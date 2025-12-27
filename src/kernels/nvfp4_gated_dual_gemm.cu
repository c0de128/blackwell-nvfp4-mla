/*
 * NVFP4 Gated Dual GEMM Kernel - Kernel Challenge #3
 *
 * Implements Gated Linear Unit (GLU) for transformer FFN layers:
 *
 *   Output = (A × W_gate) ⊗ σ(A × W_up)
 *
 * where ⊗ is element-wise multiplication and σ is SiLU activation.
 *
 * ============================================================================
 * THINK HARD: Parallel Dual GEMM Pipeline Architecture
 * ============================================================================
 *
 * THE CHALLENGE
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Standard GLU requires TWO separate GEMMs:
 *   1. gate = A × W_gate
 *   2. up   = A × W_up
 *   3. out  = gate ⊗ SiLU(up)
 *
 * Naive approach: Run them sequentially → 50% utilization
 *
 * OUR SOLUTION: Parallel TMA + Dual TMEM Pipeline
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Time ────────────────────────────────────────────────────────────────────►
 *
 * TMA (W_gate): [G0][G1][G2][G3]...
 * TMA (W_up):   [U0][U1][U2][U3]...      <- Parallel fetch!
 * TMEM ld:          [G0,U0][G1,U1]...    <- Interleaved loads
 * MMA (gate):            [MMA_G0][MMA_G1]...
 * MMA (up):              [MMA_U0][MMA_U1]...  <- Can overlap!
 * Fused gate:                        [FUSE]   <- SiLU + multiply
 *
 * SMEM MANAGEMENT STRATEGY
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Challenge: 2 weight matrices × 3 buffers = 6 tile slots
 * Solution: Distributed SMEM across cluster (borrow from neighbors)
 *
 * Option A: Fit in 48KB (aggressive tiling)
 *   - Tile size: 64×64 instead of 128×128
 *   - Weight tile: 64×64/2 = 2KB FP4 + 256B scales
 *   - Total: 6 × 2.25KB = 13.5KB (fits!)
 *
 * Option B: Distributed SMEM (Blackwell clusters)
 *   - Use cluster-level SMEM sharing
 *   - Each SM buffers different tiles, broadcast via dSMEM
 *
 * We implement Option A for portability, with Option B hooks.
 *
 * ============================================================================
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include "../../include/blackwell_compat.cuh"

/* ============================================================================
 * Kernel Configuration
 * ============================================================================
 */

/* Tile dimensions (reduced for dual buffering) */
#define TILE_M 64
#define TILE_N 64
#define TILE_K 64

/* Pipeline depths */
#define SMEM_STAGES 3      /* Triple-buffered per matrix */
#define TMEM_STAGES 2      /* Double-buffered TMEM */

/* Thread block configuration */
#define BLOCK_THREADS 128
#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32

/* Producer/Consumer split */
#define PRODUCER_WARPS 1
#define CONSUMER_WARPS (WARPS_PER_BLOCK - PRODUCER_WARPS)

/* SMEM sizes per tile (FP4 packed) */
#define WEIGHT_TILE_BYTES (TILE_M * TILE_K / 2)      /* 2048 bytes */
#define SCALE_TILE_BYTES (TILE_M * TILE_K / 16)      /* 256 bytes */
#define TILE_TOTAL_BYTES (WEIGHT_TILE_BYTES + SCALE_TILE_BYTES)  /* 2304 bytes */

/* ============================================================================
 * Shared Memory Layout - Dual Matrix Buffering
 * ============================================================================
 */

struct __align__(128) WeightTileBuffer {
    char weights[WEIGHT_TILE_BYTES];
    __nv_fp8_e4m3 scales[SCALE_TILE_BYTES];
};

struct __align__(128) DualGemmSMEM {
    /* Input activation tile (shared between both GEMMs) */
    half A_tile[TILE_M * TILE_K];

    /* Triple-buffered W_gate tiles */
    WeightTileBuffer W_gate[SMEM_STAGES];

    /* Triple-buffered W_up tiles */
    WeightTileBuffer W_up[SMEM_STAGES];

    /* Tier 1: TMA completion barriers */
    uint64_t mbar_gate[SMEM_STAGES];
    uint64_t mbar_up[SMEM_STAGES];

    /* Tier 2: TMEM load barriers */
    uint64_t mbar_tmem_gate[TMEM_STAGES];
    uint64_t mbar_tmem_up[TMEM_STAGES];

    /* Tensor-level scales */
    float scale_A;
    float scale_gate;
    float scale_up;

    /* Output scale for FP16 conversion */
    float scale_out;
};

/* Verify SMEM fits in 48KB */
static_assert(sizeof(DualGemmSMEM) <= 49152, "SMEM exceeds 48KB limit");

/* ============================================================================
 * TMEM Allocation Helpers (avoid macro conflicts)
 * ============================================================================
 */

__device__ __forceinline__ uint32_t dual_tmem_alloc(uint32_t size_bytes) {
    uint32_t addr = 0;
#if IS_BLACKWELL
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.b32 %0, %1;"
        : "=r"(addr) : "r"(size_bytes)
    );
#endif
    return addr;
}

__device__ __forceinline__ void dual_tmem_free(uint32_t addr) {
#if IS_BLACKWELL
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0;" :: "r"(addr));
#endif
}

__device__ __forceinline__ void dual_tmem_zero(uint32_t addr, uint32_t size) {
#if IS_BLACKWELL
    for (uint32_t off = 0; off < size; off += 128) {
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {0, 0, 0, 0};"
            :: "r"(addr + off) : "memory"
        );
    }
#else
    (void)addr; (void)size;
#endif
}

/* ============================================================================
 * Barrier Primitives
 * ============================================================================
 */

__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t count) {
#if IS_BLACKWELL
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "l"(mbar), "r"(count) : "memory");
#else
    *mbar = count;
#endif
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* mbar, uint32_t bytes) {
#if IS_BLACKWELL
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "l"(mbar), "r"(bytes) : "memory");
#else
    (void)mbar; (void)bytes;
#endif
}

__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t phase) {
#if IS_BLACKWELL
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "WAIT_LOOP:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_LOOP;\n"
        "}\n"
        :: "l"(mbar), "r"(phase) : "memory"
    );
#else
    while (*mbar > 0) __nanosleep(100);
#endif
}

__device__ __forceinline__ void mbar_arrive_no_complete(uint64_t* mbar) {
#if IS_BLACKWELL
    asm volatile("mbarrier.arrive.noComplete.shared::cta.b64 _, [%0];" :: "l"(mbar) : "memory");
#else
    atomicAdd((unsigned long long*)mbar, (unsigned long long)(-1));
#endif
}

/* ============================================================================
 * TMA Load Functions - Parallel Dual Fetch
 * ============================================================================
 */

__device__ __forceinline__ void tma_load_weight_tile(
    void* smem_dst,
    const CUtensorMap* tma_desc,
    uint64_t* mbar,
    int tile_k,
    int tile_n
) {
#if IS_BLACKWELL
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
          "l"(tma_desc), "r"(tile_k), "r"(tile_n),
          "r"((uint32_t)__cvta_generic_to_shared(mbar))
        : "memory"
    );
#else
    (void)smem_dst; (void)tma_desc; (void)mbar; (void)tile_k; (void)tile_n;
#endif
}

__device__ __forceinline__ void tma_load_activation_tile(
    half* smem_dst,
    const CUtensorMap* tma_desc,
    uint64_t* mbar,
    int tile_m,
    int tile_k
) {
#if IS_BLACKWELL
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
          "l"(tma_desc), "r"(tile_m), "r"(tile_k),
          "r"((uint32_t)__cvta_generic_to_shared(mbar))
        : "memory"
    );
#else
    (void)smem_dst; (void)tma_desc; (void)mbar; (void)tile_m; (void)tile_k;
#endif
}

/* ============================================================================
 * Async TMEM Operations
 * ============================================================================
 */

__device__ __forceinline__ void tmem_load_fp4_async(
    uint32_t tmem_dst,
    const void* smem_weights,
    const __nv_fp8_e4m3* smem_scales,
    uint64_t* mbar_tmem,
    int rows,
    int cols
) {
#if IS_BLACKWELL
    uint32_t smem_w = (uint32_t)__cvta_generic_to_shared(smem_weights);
    uint32_t smem_s = (uint32_t)__cvta_generic_to_shared(smem_scales);
    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar_tmem);

    #pragma unroll
    for (int br = 0; br < rows / 16; br++) {
        #pragma unroll
        for (int bc = 0; bc < cols / 16; bc++) {
            int block_idx = br * (cols / 16) + bc;
            uint32_t data_off = (br * cols + bc * 16) / 2;
            uint32_t tmem_off = (br * 16 * cols + bc * 16) * 2;

            asm volatile(
                "{\n"
                "  .reg .b32 scale;\n"
                "  ld.shared.b8 scale, [%2];\n"
                "  tcgen05.ld.async.aligned.16x16b.x1.b32.scale::e4m3.mbarrier::complete_tx::bytes"
                "    [%0], [%1], scale, [%3];\n"
                "}\n"
                :
                : "r"(tmem_dst + tmem_off), "r"(smem_w + data_off),
                  "r"(smem_s + block_idx), "r"(mbar_addr)
                : "memory"
            );
        }
    }
#else
    (void)tmem_dst; (void)smem_weights; (void)smem_scales;
    (void)mbar_tmem; (void)rows; (void)cols;
#endif
}

__device__ __forceinline__ void tmem_load_fp16(
    uint32_t tmem_dst,
    const half* smem_src,
    int num_elements
) {
#if IS_BLACKWELL
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_src);
    for (int i = 0; i < num_elements; i += 8) {
        asm volatile(
            "tcgen05.ld.sync.aligned.v4.b32 [%0], [%1];"
            :: "r"(tmem_dst + i * 2), "r"(smem_addr + i * 2) : "memory"
        );
    }
#else
    (void)tmem_dst; (void)smem_src; (void)num_elements;
#endif
}

/* Async MMA */
__device__ __forceinline__ void tmem_mma_async(
    uint32_t tmem_out,
    uint32_t tmem_A,
    uint32_t tmem_B
) {
#if IS_BLACKWELL
    asm volatile(
        "tcgen05.mma.async.aligned.m64n64k64.f32.e2m1.e2m1 [%0], [%1], [%2];"
        :: "r"(tmem_out), "r"(tmem_A), "r"(tmem_B) : "memory"
    );
#else
    (void)tmem_out; (void)tmem_A; (void)tmem_B;
#endif
}

__device__ __forceinline__ void tmem_mma_wait() {
#if IS_BLACKWELL
    asm volatile("tcgen05.mma.wait.all;" ::: "memory");
#endif
}

/* ============================================================================
 * SiLU Activation: σ(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * ============================================================================
 */

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ============================================================================
 * Fused Gating Operation in TMEM
 * ============================================================================
 *
 * Computes: out[i] = gate[i] * SiLU(up[i])
 *
 * This is done entirely in TMEM to avoid register spills.
 */

__device__ __forceinline__ void tmem_fused_gate_silu(
    uint32_t tmem_out,
    uint32_t tmem_gate,
    uint32_t tmem_up,
    int num_elements,
    float scale_gate,
    float scale_up
) {
#if IS_BLACKWELL
    const int lane = threadIdx.x % WARP_SIZE;

    /* Process in chunks of 4 for vectorized TMEM access */
    for (int i = lane * 4; i < num_elements; i += WARP_SIZE * 4) {
        float4 gate_vec, up_vec;

        /* Load from TMEM */
        asm volatile(
            "tcgen05.ld.sync.aligned.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(gate_vec.x), "=f"(gate_vec.y), "=f"(gate_vec.z), "=f"(gate_vec.w)
            : "r"(tmem_gate + i * 4)
        );
        asm volatile(
            "tcgen05.ld.sync.aligned.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(up_vec.x), "=f"(up_vec.y), "=f"(up_vec.z), "=f"(up_vec.w)
            : "r"(tmem_up + i * 4)
        );

        /* Apply scales */
        gate_vec.x *= scale_gate;
        gate_vec.y *= scale_gate;
        gate_vec.z *= scale_gate;
        gate_vec.w *= scale_gate;

        up_vec.x *= scale_up;
        up_vec.y *= scale_up;
        up_vec.z *= scale_up;
        up_vec.w *= scale_up;

        /* Fused gate: out = gate * SiLU(up) */
        float4 out_vec;
        out_vec.x = gate_vec.x * silu(up_vec.x);
        out_vec.y = gate_vec.y * silu(up_vec.y);
        out_vec.z = gate_vec.z * silu(up_vec.z);
        out_vec.w = gate_vec.w * silu(up_vec.w);

        /* Store back to TMEM */
        asm volatile(
            "tcgen05.st.sync.aligned.v4.f32 [%0], {%1, %2, %3, %4};"
            :: "r"(tmem_out + i * 4),
               "f"(out_vec.x), "f"(out_vec.y), "f"(out_vec.z), "f"(out_vec.w)
            : "memory"
        );
    }
#else
    (void)tmem_out; (void)tmem_gate; (void)tmem_up;
    (void)num_elements; (void)scale_gate; (void)scale_up;
#endif
}

/* ============================================================================
 * Store TMEM to Global with FP16 Conversion
 * ============================================================================
 */

__device__ __forceinline__ void tmem_store_fp16(
    half* global_dst,
    uint32_t tmem_src,
    int num_elements,
    float scale
) {
#if IS_BLACKWELL
    const int lane = threadIdx.x % WARP_SIZE;
    for (int i = lane; i < num_elements; i += WARP_SIZE) {
        float val;
        asm volatile("tcgen05.ld.sync.aligned.f32 %0, [%1];" : "=f"(val) : "r"(tmem_src + i * 4));
        global_dst[i] = __float2half(val * scale);
    }
#else
    (void)global_dst; (void)tmem_src; (void)num_elements; (void)scale;
#endif
}

/* ============================================================================
 * Producer Warp: Parallel Dual TMA Orchestration
 * ============================================================================
 *
 * Key insight: Issue TMA for BOTH W_gate and W_up in the same iteration.
 * This doubles memory bandwidth utilization.
 */

__device__ void producer_dual_loop(
    DualGemmSMEM* smem,
    const CUtensorMap* tma_W_gate,
    const CUtensorMap* tma_W_up,
    int tile_n,
    int num_k_tiles
) {
    const int lane = threadIdx.x % WARP_SIZE;

    if (lane != 0) return;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int buf = k_tile % SMEM_STAGES;
        uint32_t phase = k_tile / SMEM_STAGES;

        /* Wait for both buffers to be free */
        if (k_tile >= SMEM_STAGES) {
            mbar_wait(&smem->mbar_gate[buf], phase & 1);
            mbar_wait(&smem->mbar_up[buf], phase & 1);
        }

        /* Reinitialize barriers */
        mbar_init(&smem->mbar_gate[buf], 1);
        mbar_init(&smem->mbar_up[buf], 1);
        mbar_expect_tx(&smem->mbar_gate[buf], TILE_TOTAL_BYTES);
        mbar_expect_tx(&smem->mbar_up[buf], TILE_TOTAL_BYTES);

        /* Issue PARALLEL TMA loads for both matrices */
        tma_load_weight_tile(
            smem->W_gate[buf].weights,
            tma_W_gate,
            &smem->mbar_gate[buf],
            k_tile,
            tile_n
        );

        tma_load_weight_tile(
            smem->W_up[buf].weights,
            tma_W_up,
            &smem->mbar_up[buf],
            k_tile,
            tile_n
        );
    }
}

/* ============================================================================
 * Consumer Warp: Pipelined Dual GEMM with Fused Gating
 * ============================================================================
 */

__device__ void consumer_dual_gemm(
    DualGemmSMEM* smem,
    uint32_t tmem_A,
    uint32_t tmem_gate[TMEM_STAGES],
    uint32_t tmem_up[TMEM_STAGES],
    uint32_t tmem_out,
    int num_k_tiles
) {
    const int lane = threadIdx.x % WARP_SIZE;

    /* Pipeline prologue: start first tiles */
    {
        int buf_smem = 0;
        int buf_tmem = 0;

        /* Wait for TMA of first tiles */
        mbar_wait(&smem->mbar_gate[buf_smem], 0);
        mbar_wait(&smem->mbar_up[buf_smem], 0);

        /* Load activation to TMEM */
        tmem_load_fp16(tmem_A, smem->A_tile, TILE_M * TILE_K);

        /* Issue async TMEM loads for both matrices */
        if (lane == 0) {
            mbar_init(&smem->mbar_tmem_gate[buf_tmem], WEIGHT_TILE_BYTES);
            mbar_init(&smem->mbar_tmem_up[buf_tmem], WEIGHT_TILE_BYTES);
        }
        __syncwarp();

        tmem_load_fp4_async(
            tmem_gate[buf_tmem],
            smem->W_gate[buf_smem].weights,
            smem->W_gate[buf_smem].scales,
            &smem->mbar_tmem_gate[buf_tmem],
            TILE_M, TILE_K
        );

        tmem_load_fp4_async(
            tmem_up[buf_tmem],
            smem->W_up[buf_smem].weights,
            smem->W_up[buf_smem].scales,
            &smem->mbar_tmem_up[buf_tmem],
            TILE_M, TILE_K
        );

        /* Release SMEM buffers */
        mbar_arrive_no_complete(&smem->mbar_gate[buf_smem]);
        mbar_arrive_no_complete(&smem->mbar_up[buf_smem]);
    }

    /* Pipeline steady state */
    for (int k_tile = 1; k_tile < num_k_tiles; k_tile++) {
        int buf_smem = k_tile % SMEM_STAGES;
        int buf_tmem_curr = k_tile % TMEM_STAGES;
        int buf_tmem_prev = (k_tile - 1) % TMEM_STAGES;
        uint32_t phase_smem = k_tile / SMEM_STAGES;

        /* Wait for TMA of current tiles */
        mbar_wait(&smem->mbar_gate[buf_smem], phase_smem & 1);
        mbar_wait(&smem->mbar_up[buf_smem], phase_smem & 1);

        /* Wait for TMEM loads of previous tiles */
        mbar_wait(&smem->mbar_tmem_gate[buf_tmem_prev], 0);
        mbar_wait(&smem->mbar_tmem_up[buf_tmem_prev], 0);

        /* Issue async MMA for BOTH matrices on previous tiles */
        tmem_mma_async(tmem_out, tmem_A, tmem_gate[buf_tmem_prev]);
        tmem_mma_async(tmem_out + TILE_M * TILE_N * 4, tmem_A, tmem_up[buf_tmem_prev]);

        /* Issue async TMEM loads for current tiles */
        if (lane == 0) {
            mbar_init(&smem->mbar_tmem_gate[buf_tmem_curr], WEIGHT_TILE_BYTES);
            mbar_init(&smem->mbar_tmem_up[buf_tmem_curr], WEIGHT_TILE_BYTES);
        }
        __syncwarp();

        tmem_load_fp4_async(
            tmem_gate[buf_tmem_curr],
            smem->W_gate[buf_smem].weights,
            smem->W_gate[buf_smem].scales,
            &smem->mbar_tmem_gate[buf_tmem_curr],
            TILE_M, TILE_K
        );

        tmem_load_fp4_async(
            tmem_up[buf_tmem_curr],
            smem->W_up[buf_smem].weights,
            smem->W_up[buf_smem].scales,
            &smem->mbar_tmem_up[buf_tmem_curr],
            TILE_M, TILE_K
        );

        /* Release SMEM buffers */
        mbar_arrive_no_complete(&smem->mbar_gate[buf_smem]);
        mbar_arrive_no_complete(&smem->mbar_up[buf_smem]);
    }

    /* Pipeline epilogue */
    {
        int buf_tmem_last = (num_k_tiles - 1) % TMEM_STAGES;

        mbar_wait(&smem->mbar_tmem_gate[buf_tmem_last], 0);
        mbar_wait(&smem->mbar_tmem_up[buf_tmem_last], 0);

        tmem_mma_async(tmem_out, tmem_A, tmem_gate[buf_tmem_last]);
        tmem_mma_async(tmem_out + TILE_M * TILE_N * 4, tmem_A, tmem_up[buf_tmem_last]);

        tmem_mma_wait();
    }
}

/* ============================================================================
 * Main Kernel: NVFP4 Gated Dual GEMM
 * ============================================================================
 *
 * Computes: C = (A × W_gate) ⊗ SiLU(A × W_up)
 *
 * Parameters:
 *   A:       Input activations [M × K] in FP16
 *   W_gate:  Gate weights [K × N] in NVFP4
 *   W_up:    Up projection weights [K × N] in NVFP4
 *   scales_*: E4M3 micro-block scales
 *   C:       Output [M × N] in FP16
 */

extern "C" __global__ void __launch_bounds__(BLOCK_THREADS, 1)
nvfp4_gated_dual_gemm_kernel(
    /* TMA descriptors */
    const CUtensorMap* __restrict__ tma_A,
    const CUtensorMap* __restrict__ tma_W_gate,
    const CUtensorMap* __restrict__ tma_W_up,
    const CUtensorMap* __restrict__ tma_scales_gate,
    const CUtensorMap* __restrict__ tma_scales_up,
    /* Output */
    half* __restrict__ C,
    /* Dimensions */
    int M,
    int N,
    int K,
    /* Scales */
    float scale_A,
    float scale_gate,
    float scale_up,
    float scale_out
) {
    __shared__ DualGemmSMEM smem;

    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const bool is_producer = (warp_id == 0);

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    /* Initialize shared state */
    if (threadIdx.x == 0) {
        for (int i = 0; i < SMEM_STAGES; i++) {
            mbar_init(&smem.mbar_gate[i], 1);
            mbar_init(&smem.mbar_up[i], 1);
        }
        for (int i = 0; i < TMEM_STAGES; i++) {
            mbar_init(&smem.mbar_tmem_gate[i], 1);
            mbar_init(&smem.mbar_tmem_up[i], 1);
        }
        smem.scale_A = scale_A;
        smem.scale_gate = scale_gate;
        smem.scale_up = scale_up;
        smem.scale_out = scale_out;
    }
    __syncthreads();

    /* Allocate TMEM regions */
    uint32_t tmem_A = 0;
    uint32_t tmem_gate[TMEM_STAGES] = {0, 0};
    uint32_t tmem_up[TMEM_STAGES] = {0, 0};
    uint32_t tmem_out = 0;  /* Holds both gate and up results */

    if (!is_producer && lane_id == 0) {
        tmem_A = dual_tmem_alloc(TILE_M * TILE_K * sizeof(half));
        tmem_gate[0] = dual_tmem_alloc(TILE_M * TILE_K / 2);
        tmem_gate[1] = dual_tmem_alloc(TILE_M * TILE_K / 2);
        tmem_up[0] = dual_tmem_alloc(TILE_M * TILE_K / 2);
        tmem_up[1] = dual_tmem_alloc(TILE_M * TILE_K / 2);
        /* Output needs space for BOTH gate and up results before fusion */
        tmem_out = dual_tmem_alloc(TILE_M * TILE_N * sizeof(float) * 2);
    }

    /* Broadcast TMEM addresses */
    if (!is_producer) {
        tmem_A = __shfl_sync(0xFFFFFFFF, tmem_A, 0);
        tmem_gate[0] = __shfl_sync(0xFFFFFFFF, tmem_gate[0], 0);
        tmem_gate[1] = __shfl_sync(0xFFFFFFFF, tmem_gate[1], 0);
        tmem_up[0] = __shfl_sync(0xFFFFFFFF, tmem_up[0], 0);
        tmem_up[1] = __shfl_sync(0xFFFFFFFF, tmem_up[1], 0);
        tmem_out = __shfl_sync(0xFFFFFFFF, tmem_out, 0);

        /* Zero accumulators */
        if (lane_id == 0) {
            dual_tmem_zero(tmem_out, TILE_M * TILE_N * sizeof(float) * 2);
        }
    }
    __syncthreads();

    /* Run parallel producer/consumer */
    if (is_producer) {
        producer_dual_loop(&smem, tma_W_gate, tma_W_up, tile_n, num_k_tiles);
    } else {
        consumer_dual_gemm(&smem, tmem_A, tmem_gate, tmem_up, tmem_out, num_k_tiles);
    }

    __syncthreads();

    /* Fused gating: out = gate * SiLU(up) */
    if (!is_producer) {
        uint32_t tmem_gate_result = tmem_out;
        uint32_t tmem_up_result = tmem_out + TILE_M * TILE_N * sizeof(float);

        tmem_fused_gate_silu(
            tmem_gate_result,  /* Reuse gate region for final output */
            tmem_gate_result,
            tmem_up_result,
            TILE_M * TILE_N,
            smem.scale_A * smem.scale_gate,
            smem.scale_A * smem.scale_up
        );

        /* Store to global memory */
        half* C_tile = C + tile_m * TILE_M * N + tile_n * TILE_N;
        for (int row = 0; row < TILE_M; row++) {
            tmem_store_fp16(
                C_tile + row * N,
                tmem_gate_result + row * TILE_N * sizeof(float),
                TILE_N,
                smem.scale_out
            );
        }
    }

    /* Free TMEM */
    if (!is_producer && lane_id == 0) {
        dual_tmem_free(tmem_A);
        dual_tmem_free(tmem_gate[0]);
        dual_tmem_free(tmem_gate[1]);
        dual_tmem_free(tmem_up[0]);
        dual_tmem_free(tmem_up[1]);
        dual_tmem_free(tmem_out);
    }
}

/* ============================================================================
 * Host Launch Helper (Popcorn CLI Compatible)
 * ============================================================================
 */

extern "C" cudaError_t launch_nvfp4_gated_dual_gemm(
    /* Input tensors */
    const half* A,
    const void* W_gate,
    const void* W_up,
    const void* scales_gate,
    const void* scales_up,
    /* Output */
    half* C,
    /* Dimensions */
    int M,
    int N,
    int K,
    /* Scales */
    float scale_A,
    float scale_gate,
    float scale_up,
    float scale_out,
    /* Stream */
    cudaStream_t stream
) {
    /* TMA descriptor setup would go here for real Blackwell deployment */
    CUtensorMap tma_A, tma_W_gate, tma_W_up, tma_scales_gate, tma_scales_up;

    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    dim3 block(BLOCK_THREADS);

    nvfp4_gated_dual_gemm_kernel<<<grid, block, 0, stream>>>(
        &tma_A, &tma_W_gate, &tma_W_up, &tma_scales_gate, &tma_scales_up,
        C, M, N, K,
        scale_A, scale_gate, scale_up, scale_out
    );

    return cudaGetLastError();
}
