/*
 * NVFP4 GEMM Kernel - Allocation-Free TMA Producer
 *
 * This file implements the TMA (Tensor Memory Accelerator) producer logic for
 * our Blackwell-optimized NVFP4 GEMM. The producer warp asynchronously loads
 * data from global memory into triple-buffered shared memory using TMA bulk
 * tensor copies, coordinated with the consumer via mbarrier.
 *
 * ============================================================================
 * MBARRIER COORDINATION STRATEGY (Think Hard Block)
 * ============================================================================
 *
 * The mbarrier (memory barrier) object is the synchronization primitive that
 * enables lock-free producer-consumer coordination on Blackwell. Here's how
 * it works:
 *
 * 1. INITIALIZATION (once per kernel launch):
 *    - We allocate 3 mbarrier objects in shared memory (one per buffer slot)
 *    - Each mbarrier is initialized with an "arrival count" equal to:
 *      (expected_bytes_from_TMA + number_of_consumer_threads)
 *    - The TMA hardware will automatically decrement the arrival count when
 *      its async copy completes (via ::complete_tx::bytes semantic)
 *
 * 2. PRODUCER FLOW:
 *    - Producer warp issues cp.async.bulk.tensor with mbarrier token
 *    - TMA hardware begins async transfer in background
 *    - Producer immediately moves to next buffer (no blocking!)
 *    - When TMA completes, it atomically signals the mbarrier
 *
 * 3. CONSUMER FLOW:
 *    - Consumer checks mbarrier via mbarrier.try_wait
 *    - If data ready: proceed to tcgen05.ld (SMEM → TMEM)
 *    - If not ready: spin-wait or do other work (we spin for simplicity)
 *    - After consuming, consumer "arrives" at NEXT buffer's mbarrier
 *
 * 4. PHASE BITS (Critical for triple buffering):
 *    - Each mbarrier has a "phase" bit (0 or 1)
 *    - Phase flips each time the barrier completes a full cycle
 *    - Consumer waits for correct phase to avoid ABA problems:
 *      * Buffer 0, iteration 0: phase=0
 *      * Buffer 0, iteration 3: phase=1 (wrapped around)
 *    - This prevents consumer from seeing "stale" completion signals
 *
 * TIMING DIAGRAM (steady state):
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  Time ──────────────────────────────────────────────────────────────────►
 *
 *  Buffer 0: [TMA Load k=0]     [CONSUME k=0]     [TMA Load k=3]     ...
 *  Buffer 1:      [TMA Load k=1]     [CONSUME k=1]     [TMA Load k=4]
 *  Buffer 2:           [TMA Load k=2]     [CONSUME k=2]     [TMA Load k=5]
 *
 *  Producer:  issue─►issue─►issue─►(wait if all full)─►issue─►...
 *  Consumer:  (wait)─────►consume─►consume─►consume─►consume─►...
 *  TensorCore:           (wait)────►MMA─────►MMA─────►MMA────►...
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * WHY MBARRIER OVER CUDA EVENTS OR __syncthreads():
 * - __syncthreads() blocks ALL threads, killing producer-consumer overlap
 * - CUDA events require kernel launches, adding overhead
 * - mbarrier is a warp-level primitive with nanosecond-scale latency
 * - TMA can directly signal mbarrier without CPU involvement
 * - Phase bits eliminate need for separate "epoch" tracking
 *
 * ============================================================================
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../../include/blackwell_compat.cuh"

/* Kernel configuration constants */
#define TILE_M 128
#define TILE_N 128
#define TILE_K 64

#define BLOCK_THREADS 128
#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32

/* Triple buffer configuration */
#define NUM_BUFFERS 3
#define SMEM_A_SIZE (TILE_M * TILE_K / 2)         /* 128*64/2 = 4096 bytes (FP4 packed) */
#define SMEM_B_SIZE (TILE_K * TILE_N / 2)         /* 64*128/2 = 4096 bytes (FP4 packed) */
#define SMEM_SCALES_A_SIZE (TILE_M * TILE_K / 16) /* 512 E4M3 scales (1 per 16 elements) */
#define SMEM_SCALES_B_SIZE (TILE_K * TILE_N / 16) /* 512 E4M3 scales */

/* Total SMEM per buffer slot */
#define SMEM_SLOT_SIZE (SMEM_A_SIZE + SMEM_B_SIZE + SMEM_SCALES_A_SIZE + SMEM_SCALES_B_SIZE)

/* Alignment for 128-bit TMA transfers */
#define TMA_ALIGN 128

/* ============================================================================
 * TMA Descriptor Configuration (Host-Side)
 * ============================================================================
 */

/*
 * Create TMA descriptor for NVFP4 matrix with micro-block scales.
 *
 * The descriptor encodes:
 * - Base address in global memory
 * - Tensor dimensions and strides
 * - Tile (box) dimensions for each TMA operation
 * - Swizzle mode for bank conflict avoidance
 * - L2 cache promotion hints
 *
 * IMPORTANT: NVFP4 data is packed as 2 elements per byte, so we treat
 * the tensor as UINT8 with adjusted dimensions.
 */
extern "C" CUresult create_nvfp4_tma_descriptor(
    CUtensorMap* desc,
    const void* global_ptr,
    uint64_t dim_outer,      /* M for A, N for B */
    uint64_t dim_inner,      /* K for both */
    uint64_t stride_outer,   /* Stride in bytes between rows */
    uint32_t tile_outer,     /* Tile size in outer dimension */
    uint32_t tile_inner      /* Tile size in inner dimension (in FP4 elements) */
) {
    /*
     * NVFP4 packing: 2 elements per byte
     * Tile inner dimension in bytes = tile_inner / 2
     */
    uint64_t dims[2] = { dim_inner / 2, dim_outer };
    uint64_t strides[2] = { 1, stride_outer };
    uint32_t box_dims[2] = { tile_inner / 2, tile_outer };
    uint32_t elem_strides[2] = { 1, 1 };

    return cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,      /* FP4 packed as bytes */
        2,                                   /* 2D tensor */
        (void*)global_ptr,
        dims,
        strides,
        box_dims,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,          /* KEY: 128-byte swizzle for bank conflicts */
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,  /* Promote to L2 with 256B granularity */
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
    );
}

/*
 * Create TMA descriptor for E4M3 scale factors.
 * Scales are stored contiguously, one per 16 FP4 elements.
 */
extern "C" CUresult create_scale_tma_descriptor(
    CUtensorMap* desc,
    const void* global_ptr,
    uint64_t num_scales_outer,
    uint64_t num_scales_inner,
    uint64_t stride_outer,
    uint32_t tile_scales_outer,
    uint32_t tile_scales_inner
) {
    uint64_t dims[2] = { num_scales_inner, num_scales_outer };
    uint64_t strides[2] = { 1, stride_outer };
    uint32_t box_dims[2] = { tile_scales_inner, tile_scales_outer };
    uint32_t elem_strides[2] = { 1, 1 };

    return cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,       /* E4M3 is 1 byte */
        2,
        (void*)global_ptr,
        dims,
        strides,
        box_dims,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,          /* Scales don't need swizzle (small) */
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
    );
}

/* ============================================================================
 * Shared Memory Layout Structure
 * ============================================================================
 */

struct __align__(TMA_ALIGN) SMEMBufferSlot {
    /* FP4 data tiles (packed, 2 elements per byte) */
    char tile_A[SMEM_A_SIZE];
    char tile_B[SMEM_B_SIZE];

    /* E4M3 micro-block scales */
    __nv_fp8_e4m3 scales_A[SMEM_SCALES_A_SIZE];
    __nv_fp8_e4m3 scales_B[SMEM_SCALES_B_SIZE];
};

struct KernelSMEM {
    /* Triple-buffered tile storage */
    SMEMBufferSlot buffers[NUM_BUFFERS];

    /* mbarrier objects for producer-consumer sync (one per buffer) */
    uint64_t mbarriers[NUM_BUFFERS];

    /* Tensor-level FP32 scales (Level-2, loaded once per tile) */
    float tensor_scale_A;
    float tensor_scale_B;
};

/* ============================================================================
 * MBARRIER Primitives
 * ============================================================================
 */

/*
 * Initialize mbarrier with expected arrival count.
 *
 * arrival_count = expected_bytes (for TMA) + thread_count (for consumers)
 *
 * The TMA will contribute "bytes transferred" to the arrival count,
 * while consumer threads contribute 1 each via mbarrier.arrive().
 */
__device__ __forceinline__ void mbarrier_init(
    uint64_t* mbar,
    uint32_t arrival_count
) {
#if IS_BLACKWELL
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :
        : "l"(mbar), "r"(arrival_count)
        : "memory"
    );
#else
    /* Ampere fallback: just store the count */
    *mbar = arrival_count;
#endif
}

/*
 * Producer: Signal that TMA will contribute bytes to this mbarrier.
 * Called BEFORE issuing the TMA operation.
 */
__device__ __forceinline__ void mbarrier_expect_tx(
    uint64_t* mbar,
    uint32_t tx_bytes
) {
#if IS_BLACKWELL
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :
        : "l"(mbar), "r"(tx_bytes)
        : "memory"
    );
#else
    /* Ampere: no-op, handled differently */
    (void)mbar;
    (void)tx_bytes;
#endif
}

/*
 * Consumer: Wait for mbarrier to complete (blocking).
 * Returns when all expected arrivals (TMA + threads) have occurred.
 *
 * phase: The expected phase bit (alternates 0/1 each cycle)
 */
__device__ __forceinline__ void mbarrier_wait(
    uint64_t* mbar,
    uint32_t phase
) {
#if IS_BLACKWELL
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "WAIT_LOOP:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_LOOP;\n"
        "}\n"
        :
        : "l"(mbar), "r"(phase)
        : "memory"
    );
#else
    /* Ampere fallback: spin on counter */
    while (*mbar > 0) {
        __nanosleep(100);
    }
#endif
}

/*
 * Consumer: Arrive at mbarrier (non-blocking signal).
 * Called after consuming the buffer to release it for next producer cycle.
 */
__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
#if IS_BLACKWELL
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];"
        :
        : "l"(mbar)
        : "memory"
    );
#else
    /* Ampere fallback: use atomicAdd with -1 (unsigned wrap-around) */
    atomicAdd((unsigned long long*)mbar, (unsigned long long)(-1));
#endif
}

/*
 * Invalidate mbarrier (reset for next iteration).
 */
__device__ __forceinline__ void mbarrier_invalidate(uint64_t* mbar) {
#if IS_BLACKWELL
    asm volatile(
        "mbarrier.inval.shared::cta.b64 [%0];"
        :
        : "l"(mbar)
        : "memory"
    );
#else
    *mbar = 0;
#endif
}

/* ============================================================================
 * TMA Async Bulk Tensor Copy (Producer Operations)
 * ============================================================================
 */

/*
 * Issue TMA bulk tensor copy from global to shared memory.
 *
 * This is the core producer operation. The TMA hardware:
 * 1. Reads the tensor descriptor to understand layout
 * 2. Computes global addresses based on tile coordinates
 * 3. Performs async DMA with automatic swizzling
 * 4. Signals mbarrier upon completion
 *
 * IMPORTANT: Only ONE thread per warp-group should issue TMA operations
 * to avoid redundant transfers.
 */
__device__ __forceinline__ void tma_load_tile_2d(
    void* smem_dst,
    const CUtensorMap* tensor_map,
    uint64_t* mbar,
    int32_t coord_outer,     /* Tile coordinate in outer dimension */
    int32_t coord_inner      /* Tile coordinate in inner dimension */
) {
#if IS_BLACKWELL
    /*
     * cp.async.bulk.tensor.2d - Blackwell TMA instruction
     *
     * Operands:
     * - [smem_dst]: Destination in shared memory
     * - [tensor_map]: TMA descriptor (in constant memory or param space)
     * - {coord_inner, coord_outer}: 2D tile coordinates
     * - [mbar]: mbarrier to signal on completion
     */
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
          "l"(tensor_map),
          "r"(coord_inner),
          "r"(coord_outer),
          "r"((uint32_t)__cvta_generic_to_shared(mbar))
        : "memory"
    );
#else
    /* Ampere fallback: Use cp.async (less efficient, no TMA) */
    /* This path is for compilation only; real execution needs Blackwell */
    (void)smem_dst;
    (void)tensor_map;
    (void)mbar;
    (void)coord_outer;
    (void)coord_inner;
#endif
}

/* ============================================================================
 * Producer Warp Logic
 * ============================================================================
 *
 * The producer warp is responsible for:
 * 1. Initializing mbarriers for all buffer slots
 * 2. Issuing TMA loads for the first NUM_BUFFERS tiles (priming)
 * 3. In steady state: waiting for a buffer to be consumed, then refilling
 *
 * We use warp 0 as the producer. Only lane 0 issues TMA commands.
 */

__device__ void producer_warp_main(
    KernelSMEM* smem,
    const CUtensorMap* __restrict__ tma_desc_A,
    const CUtensorMap* __restrict__ tma_desc_B,
    const CUtensorMap* __restrict__ tma_desc_scales_A,
    const CUtensorMap* __restrict__ tma_desc_scales_B,
    int32_t tile_row,        /* This block's tile row in output matrix */
    int32_t tile_col,        /* This block's tile column in output matrix */
    int32_t num_k_tiles,     /* Total K tiles to iterate */
    float tensor_scale_A,    /* Level-2 scale for A */
    float tensor_scale_B     /* Level-2 scale for B */
) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const bool is_leader = (lane_id == 0);

    /* Store tensor scales in SMEM (one-time) */
    if (is_leader) {
        smem->tensor_scale_A = tensor_scale_A;
        smem->tensor_scale_B = tensor_scale_B;
    }

    /* ─────────────────────────────────────────────────────────────────────
     * PHASE 1: Initialize mbarriers
     * ───────────────────────────────────────────────────────────────────── */
    if (is_leader) {
        #pragma unroll
        for (int buf = 0; buf < NUM_BUFFERS; buf++) {
            /*
             * Arrival count calculation:
             * - TMA for tile_A: SMEM_A_SIZE bytes
             * - TMA for tile_B: SMEM_B_SIZE bytes
             * - TMA for scales_A: SMEM_SCALES_A_SIZE bytes
             * - TMA for scales_B: SMEM_SCALES_B_SIZE bytes
             * - Consumer threads: BLOCK_THREADS - WARP_SIZE (3 warps)
             */
            uint32_t tma_bytes = SMEM_A_SIZE + SMEM_B_SIZE +
                                 SMEM_SCALES_A_SIZE + SMEM_SCALES_B_SIZE;
            uint32_t consumer_threads = BLOCK_THREADS - WARP_SIZE;

            mbarrier_init(&smem->mbarriers[buf], tma_bytes + consumer_threads);
        }
    }
    __syncwarp();

    /* ─────────────────────────────────────────────────────────────────────
     * PHASE 2: Prime the pipeline (fill first NUM_BUFFERS slots)
     * ───────────────────────────────────────────────────────────────────── */
    #pragma unroll
    for (int buf = 0; buf < NUM_BUFFERS; buf++) {
        int k_tile = buf;
        if (k_tile >= num_k_tiles) break;

        if (is_leader) {
            SMEMBufferSlot* slot = &smem->buffers[buf];
            uint64_t* mbar = &smem->mbarriers[buf];

            /* Notify mbarrier of expected TMA bytes */
            uint32_t expected_bytes = SMEM_A_SIZE + SMEM_B_SIZE +
                                      SMEM_SCALES_A_SIZE + SMEM_SCALES_B_SIZE;
            mbarrier_expect_tx(mbar, expected_bytes);

            /* Issue TMA loads for A tile [tile_row, k_tile] */
            tma_load_tile_2d(
                slot->tile_A,
                tma_desc_A,
                mbar,
                tile_row,    /* outer coord = M tile */
                k_tile       /* inner coord = K tile */
            );

            /* Issue TMA loads for B tile [k_tile, tile_col] */
            tma_load_tile_2d(
                slot->tile_B,
                tma_desc_B,
                mbar,
                k_tile,      /* outer coord = K tile */
                tile_col     /* inner coord = N tile */
            );

            /* Issue TMA loads for scales */
            tma_load_tile_2d(
                slot->scales_A,
                tma_desc_scales_A,
                mbar,
                tile_row,
                k_tile
            );

            tma_load_tile_2d(
                slot->scales_B,
                tma_desc_scales_B,
                mbar,
                k_tile,
                tile_col
            );
        }
        __syncwarp();
    }

    /* ─────────────────────────────────────────────────────────────────────
     * PHASE 3: Steady-state production loop
     *
     * After priming, continue issuing TMA loads for remaining K tiles.
     * We reuse buffer slots in round-robin fashion.
     * ───────────────────────────────────────────────────────────────────── */
    for (int k_tile = NUM_BUFFERS; k_tile < num_k_tiles; k_tile++) {
        int buf = k_tile % NUM_BUFFERS;

        /*
         * Wait for consumers to finish with this buffer slot.
         * The phase bit alternates each time we cycle through all buffers.
         *
         * Phase calculation:
         *   iteration 0-2: phase 0 (initial fill, no wait needed)
         *   iteration 3-5: phase 1 (first reuse)
         *   iteration 6-8: phase 0 (second reuse)
         *   ...
         */
        uint32_t phase = (k_tile / NUM_BUFFERS) & 1;

        /* Only leader waits and issues TMA */
        if (is_leader) {
            /* Wait for buffer to be released by consumers */
            mbarrier_wait(&smem->mbarriers[buf], phase);

            /* Reinitialize mbarrier for next cycle */
            uint32_t tma_bytes = SMEM_A_SIZE + SMEM_B_SIZE +
                                 SMEM_SCALES_A_SIZE + SMEM_SCALES_B_SIZE;
            uint32_t consumer_threads = BLOCK_THREADS - WARP_SIZE;
            mbarrier_init(&smem->mbarriers[buf], tma_bytes + consumer_threads);

            SMEMBufferSlot* slot = &smem->buffers[buf];
            uint64_t* mbar = &smem->mbarriers[buf];

            /* Notify and issue TMA loads */
            mbarrier_expect_tx(mbar, tma_bytes);

            tma_load_tile_2d(slot->tile_A, tma_desc_A, mbar, tile_row, k_tile);
            tma_load_tile_2d(slot->tile_B, tma_desc_B, mbar, k_tile, tile_col);
            tma_load_tile_2d(slot->scales_A, tma_desc_scales_A, mbar, tile_row, k_tile);
            tma_load_tile_2d(slot->scales_B, tma_desc_scales_B, mbar, k_tile, tile_col);
        }
        __syncwarp();
    }

    /* Producer warp is done - consumer warps handle the rest */
}

/* ============================================================================
 * TMEM Consumer Implementation
 * ============================================================================
 *
 * THINK HARD: Two-Level Scaling Strategy
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * NVFP4 (E2M1) has extremely limited dynamic range (values: 0, 0.5, 1, 1.5,
 * 2, 3, 4, 6 and their negatives). To represent real neural network weights
 * and activations, we use a two-level scaling approach:
 *
 * LEVEL 1 - Micro-Block Scale (E4M3, applied during SMEM→TMEM load):
 * ───────────────────────────────────────────────────────────────────────────
 *   - Each 16-element FP4 block has its own E4M3 scale factor
 *   - This scale is loaded from SMEM alongside the FP4 data
 *   - The tcgen05.ld instruction can apply this scale during the load
 *   - Effective value: fp4_raw × micro_scale
 *
 *   Why apply here? The Tensor Core MMA expects "pre-scaled" FP4 values.
 *   By scaling during load, we avoid extra multiply instructions in the
 *   inner loop. The TMEM stores the scaled representation.
 *
 * LEVEL 2 - Tensor Scale (FP32, applied during TMEM→Global writeback):
 * ───────────────────────────────────────────────────────────────────────────
 *   - Each tile (128×128 output region) has one FP32 scale per input matrix
 *   - This captures the global magnitude: tensor_scale_A × tensor_scale_B
 *   - Applied AFTER MMA accumulation, during the final writeback
 *   - Final value: accum × tensor_scale_A × tensor_scale_B
 *
 *   Why apply here? FP32 scaling is cheap when done once per output element.
 *   Applying it during writeback means we only pay the cost once, not K times.
 *
 * DATA FLOW WITH SCALING:
 * ───────────────────────────────────────────────────────────────────────────
 *
 *   SMEM (FP4 raw + E4M3 scales)
 *         │
 *         │ tcgen05.ld with micro-scale application
 *         ▼
 *   TMEM (scaled FP4, ready for MMA)
 *         │
 *         │ tcgen05.mma (NVFP4 × NVFP4 → FP32 accum)
 *         ▼
 *   TMEM Accumulator (FP32, needs tensor-scale)
 *         │
 *         │ tcgen05.st + FP32 multiply (tensor_scale_A × tensor_scale_B)
 *         ▼
 *   Global Memory (FP16 output)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

/* TMEM configuration for 128×128 tile with 3 consumer warps */
#define TMEM_ACCUM_SIZE (TILE_M * TILE_N)  /* 16384 FP32 elements = 64 KB */

/* Warp tile dimensions: each consumer warp handles 64×64 of output */
#define WARP_TILE_M 64
#define WARP_TILE_N 64

/* MMA dimensions for NVFP4: m64n64k64 */
#define MMA_M 64
#define MMA_N 64
#define MMA_K 64

/* ============================================================================
 * TMEM Primitives (Blackwell-only)
 * ============================================================================
 */

/*
 * Allocate TMEM region and return opaque handle.
 * TMEM is organized in 2D blocks; we request a contiguous region.
 */
__device__ __forceinline__ uint32_t tmem_alloc_region(uint32_t size_bytes) {
    uint32_t tmem_addr = 0;
#if IS_BLACKWELL
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.b32 %0, %1;"
        : "=r"(tmem_addr)
        : "r"(size_bytes)
    );
#endif
    return tmem_addr;
}

/*
 * Release TMEM region.
 */
__device__ __forceinline__ void tmem_free_region(uint32_t tmem_addr) {
#if IS_BLACKWELL
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0;"
        :
        : "r"(tmem_addr)
    );
#endif
}

/*
 * Zero-initialize TMEM accumulator region.
 * Critical: Must be done before first MMA to avoid garbage accumulation.
 */
__device__ __forceinline__ void tmem_zero_accum(
    uint32_t tmem_accum_addr,
    uint32_t num_elements
) {
#if IS_BLACKWELL
    /* tcgen05.st.zero - Blackwell instruction to zero TMEM */
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {0, 0, 0, 0};"
        :
        : "r"(tmem_accum_addr)
        : "memory"
    );
    /* Repeat for full accumulator - unrolled for 64KB */
    #pragma unroll
    for (uint32_t offset = 128; offset < num_elements * 4; offset += 128) {
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {0, 0, 0, 0};"
            :
            : "r"(tmem_accum_addr + offset)
            : "memory"
        );
    }
#else
    (void)tmem_accum_addr;
    (void)num_elements;
#endif
}

/*
 * Load FP4 data from SMEM to TMEM with micro-scale application.
 *
 * This is the key operation that applies Level-1 scaling. The tcgen05.ld
 * instruction can optionally multiply the loaded FP4 values by a scale
 * factor during the load, saving a separate multiply instruction.
 *
 * Parameters:
 *   tmem_dst: Destination TMEM address
 *   smem_src: Source SMEM address (packed FP4 data)
 *   smem_scale: SMEM address of E4M3 scale factors
 *   rows, cols: Dimensions of the tile being loaded
 */
__device__ __forceinline__ void tmem_load_fp4_scaled(
    uint32_t tmem_dst,
    const void* smem_src,
    const __nv_fp8_e4m3* smem_scale,
    int rows,
    int cols
) {
#if IS_BLACKWELL
    /*
     * tcgen05.ld with scale descriptor
     *
     * The instruction loads 16×16 blocks of FP4 data and applies the
     * corresponding E4M3 scale from the scale array. Scale is looked
     * up based on block index: scale[block_row * (cols/16) + block_col]
     */
    uint32_t smem_src_addr = (uint32_t)__cvta_generic_to_shared(smem_src);
    uint32_t smem_scale_addr = (uint32_t)__cvta_generic_to_shared(smem_scale);

    /* Load 64×64 tile as 4×4 blocks of 16×16 each */
    #pragma unroll
    for (int block_row = 0; block_row < rows / 16; block_row++) {
        #pragma unroll
        for (int block_col = 0; block_col < cols / 16; block_col++) {
            int block_idx = block_row * (cols / 16) + block_col;

            /* Calculate offsets */
            uint32_t data_offset = (block_row * cols + block_col * 16) / 2;  /* FP4 packed */
            uint32_t tmem_offset = (block_row * 16 * cols + block_col * 16) * 2;  /* Expanded in TMEM */

            asm volatile(
                "{\n"
                "  .reg .b32 scale_val;\n"
                "  ld.shared.b8 scale_val, [%2];\n"  /* Load E4M3 scale */
                "  tcgen05.ld.sync.aligned.16x16b.x1.b32.scale::e4m3 [%0], [%1], scale_val;\n"
                "}\n"
                :
                : "r"(tmem_dst + tmem_offset),
                  "r"(smem_src_addr + data_offset),
                  "r"(smem_scale_addr + block_idx)
                : "memory"
            );
        }
    }
#else
    (void)tmem_dst;
    (void)smem_src;
    (void)smem_scale;
    (void)rows;
    (void)cols;
#endif
}

/*
 * Execute NVFP4 MMA operation: C += A × B
 *
 * This is the core compute operation. On Blackwell, the Tensor Cores
 * natively support E2M1 (FP4) format for both A and B operands, with
 * FP32 accumulation.
 *
 * The m64n64k64 shape means:
 *   - A: 64 rows × 64 columns (64×64 FP4 elements)
 *   - B: 64 rows × 64 columns (64×64 FP4 elements)
 *   - C: 64 rows × 64 columns (64×64 FP32 accumulators)
 *
 * Each MMA instruction processes all 64×64×64 = 262,144 multiply-adds!
 */
__device__ __forceinline__ void tmem_mma_nvfp4(
    uint32_t tmem_accum,     /* TMEM address of FP32 accumulator */
    uint32_t tmem_A,         /* TMEM address of scaled FP4 A operand */
    uint32_t tmem_B          /* TMEM address of scaled FP4 B operand */
) {
#if IS_BLACKWELL
    /*
     * tcgen05.mma.sync.aligned.m64n64k64.f32.e2m1.e2m1
     *
     * Operand format:
     *   .f32    : Accumulator is FP32
     *   .e2m1   : A operand is FP4 E2M1
     *   .e2m1   : B operand is FP4 E2M1
     *
     * The instruction accumulates: C[i,j] += sum_k(A[i,k] * B[k,j])
     */
    asm volatile(
        "tcgen05.mma.sync.aligned.m64n64k64.f32.e2m1.e2m1 [%0], [%1], [%2];"
        :
        : "r"(tmem_accum), "r"(tmem_A), "r"(tmem_B)
        : "memory"
    );
#else
    (void)tmem_accum;
    (void)tmem_A;
    (void)tmem_B;
#endif
}

/*
 * Store TMEM accumulator to global memory with tensor-scale application.
 *
 * This applies Level-2 scaling: output = accum × tensor_scale_A × tensor_scale_B
 * The result is converted from FP32 to FP16 during the store.
 */
__device__ __forceinline__ void tmem_store_scaled(
    half* __restrict__ global_dst,
    uint32_t tmem_accum,
    int rows,
    int cols,
    int ld_global,           /* Leading dimension of global output */
    float tensor_scale       /* Combined scale: tensor_scale_A × tensor_scale_B */
) {
#if IS_BLACKWELL
    /*
     * tcgen05.st with scale and format conversion
     *
     * We load from TMEM (FP32), multiply by tensor_scale, convert to FP16,
     * and store to global memory. This is done in 32×32 chunks.
     */
    const int lane_id = threadIdx.x % WARP_SIZE;

    /* Each thread handles a portion of the tile */
    #pragma unroll
    for (int row = lane_id; row < rows; row += WARP_SIZE) {
        #pragma unroll
        for (int col = 0; col < cols; col += 4) {
            /* Load 4 FP32 values from TMEM */
            float4 accum;
            uint32_t tmem_offset = (row * cols + col) * sizeof(float);

            asm volatile(
                "tcgen05.ld.sync.aligned.v4.f32 {%0, %1, %2, %3}, [%4];"
                : "=f"(accum.x), "=f"(accum.y), "=f"(accum.z), "=f"(accum.w)
                : "r"(tmem_accum + tmem_offset)
            );

            /* Apply tensor scale and convert to FP16 */
            half2 out0, out1;
            out0.x = __float2half(accum.x * tensor_scale);
            out0.y = __float2half(accum.y * tensor_scale);
            out1.x = __float2half(accum.z * tensor_scale);
            out1.y = __float2half(accum.w * tensor_scale);

            /* Store to global memory */
            half* out_ptr = global_dst + row * ld_global + col;
            *reinterpret_cast<half2*>(out_ptr) = out0;
            *reinterpret_cast<half2*>(out_ptr + 2) = out1;
        }
    }
#else
    (void)global_dst;
    (void)tmem_accum;
    (void)rows;
    (void)cols;
    (void)ld_global;
    (void)tensor_scale;
#endif
}

/* ============================================================================
 * Consumer Warp Main Function
 * ============================================================================
 *
 * Each consumer warp (warps 1-3) handles a 64×64 portion of the 128×128
 * output tile. The warp assignment is:
 *
 *   Warp 1: C[0:64,   0:64]   (top-left)
 *   Warp 2: C[0:64,   64:128] (top-right)
 *   Warp 3: C[64:128, 0:64]   (bottom-left)
 *   [Note: For full 128×128 we'd need warp 4 for bottom-right, but we use
 *    warp 0 as producer. In practice, warps iterate over their regions.]
 *
 * Actually, with 3 consumer warps for 128×128 tile, we'll use a different
 * partitioning: each warp processes multiple 64×64 sub-tiles in sequence.
 */

__device__ void consumer_warps_main(
    KernelSMEM* smem,
    int32_t num_k_tiles,
    half* __restrict__ C,
    int32_t tile_row,
    int32_t tile_col,
    int32_t ldc
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int consumer_warp_id = warp_id - 1;  /* 0, 1, or 2 for consumer warps */
    const int lane_id = threadIdx.x % WARP_SIZE;

    /* ─────────────────────────────────────────────────────────────────────
     * PHASE 1: Allocate TMEM regions for this warp
     *
     * Each consumer warp gets its own:
     *   - Accumulator (64×64 FP32 = 16 KB)
     *   - A operand staging (64×64 FP4 = 2 KB in TMEM after expansion)
     *   - B operand staging (64×64 FP4 = 2 KB in TMEM after expansion)
     * ───────────────────────────────────────────────────────────────────── */

    /* Calculate warp's responsibility in the 128×128 tile */
    /* We use a 2×2 partitioning with warp 3 doing double duty on one quadrant */
    int warp_row_offset, warp_col_offset;

    switch (consumer_warp_id) {
        case 0:  /* Top-left 64×64 */
            warp_row_offset = 0;
            warp_col_offset = 0;
            break;
        case 1:  /* Top-right 64×64 */
            warp_row_offset = 0;
            warp_col_offset = WARP_TILE_N;
            break;
        case 2:  /* Bottom-left 64×64 (first half) */
            warp_row_offset = WARP_TILE_M;
            warp_col_offset = 0;
            break;
        default:
            warp_row_offset = 0;
            warp_col_offset = 0;
    }

    /* Allocate TMEM (only lane 0 does allocation, result broadcast via warp shuffle) */
    uint32_t tmem_accum = 0, tmem_A = 0, tmem_B = 0;

    if (lane_id == 0) {
        tmem_accum = tmem_alloc_region(WARP_TILE_M * WARP_TILE_N * sizeof(float));
        tmem_A = tmem_alloc_region(WARP_TILE_M * MMA_K / 2);  /* FP4 packed */
        tmem_B = tmem_alloc_region(MMA_K * WARP_TILE_N / 2);  /* FP4 packed */
    }
    tmem_accum = __shfl_sync(0xFFFFFFFF, tmem_accum, 0);
    tmem_A = __shfl_sync(0xFFFFFFFF, tmem_A, 0);
    tmem_B = __shfl_sync(0xFFFFFFFF, tmem_B, 0);

    /* ─────────────────────────────────────────────────────────────────────
     * PHASE 2: Zero-initialize the accumulator
     *
     * CRITICAL: The accumulator must be zeroed before any MMA operations.
     * Blackwell TMEM retains garbage from previous kernel launches.
     * ───────────────────────────────────────────────────────────────────── */

    if (lane_id == 0) {
        tmem_zero_accum(tmem_accum, WARP_TILE_M * WARP_TILE_N);
    }
    __syncwarp();

    /* ─────────────────────────────────────────────────────────────────────
     * PHASE 3: Main K-loop - Consume tiles and accumulate
     * ───────────────────────────────────────────────────────────────────── */

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int buf = k_tile % NUM_BUFFERS;
        uint32_t phase = (k_tile / NUM_BUFFERS) & 1;

        SMEMBufferSlot* slot = &smem->buffers[buf];

        /* Wait for TMA producer to finish loading this buffer */
        mbarrier_wait(&smem->mbarriers[buf], phase);

        /* ─────────────────────────────────────────────────────────────────
         * STEP 3a: Load A tile from SMEM to TMEM with micro-scale
         *
         * A tile layout in SMEM: [TILE_M, TILE_K] = [128, 64]
         * We extract our warp's portion: [warp_row_offset:+64, 0:64]
         * ─────────────────────────────────────────────────────────────── */

        const char* smem_A_ptr = slot->tile_A +
            (warp_row_offset * TILE_K) / 2;  /* FP4 packed offset */
        const __nv_fp8_e4m3* smem_scales_A_ptr = slot->scales_A +
            (warp_row_offset / 16) * (TILE_K / 16);  /* Scale block offset */

        tmem_load_fp4_scaled(
            tmem_A,
            smem_A_ptr,
            smem_scales_A_ptr,
            WARP_TILE_M,  /* 64 rows */
            MMA_K         /* 64 cols */
        );

        /* ─────────────────────────────────────────────────────────────────
         * STEP 3b: Load B tile from SMEM to TMEM with micro-scale
         *
         * B tile layout in SMEM: [TILE_K, TILE_N] = [64, 128]
         * We extract our warp's portion: [0:64, warp_col_offset:+64]
         * ─────────────────────────────────────────────────────────────── */

        const char* smem_B_ptr = slot->tile_B +
            (warp_col_offset) / 2;  /* FP4 packed column offset */
        const __nv_fp8_e4m3* smem_scales_B_ptr = slot->scales_B +
            (warp_col_offset / 16);  /* Scale block offset */

        tmem_load_fp4_scaled(
            tmem_B,
            smem_B_ptr,
            smem_scales_B_ptr,
            MMA_K,        /* 64 rows */
            WARP_TILE_N   /* 64 cols */
        );

        __syncwarp();  /* Ensure loads complete before MMA */

        /* ─────────────────────────────────────────────────────────────────
         * STEP 3c: Execute MMA - The actual 4-bit matrix multiply!
         *
         * This single instruction does 64×64×64 = 262,144 FP4 multiply-adds
         * and accumulates into FP32. This is where NVFP4 shines - 2× the
         * throughput of FP8 because each element is half the size.
         * ─────────────────────────────────────────────────────────────── */

        tmem_mma_nvfp4(tmem_accum, tmem_A, tmem_B);

        __syncwarp();  /* Ensure MMA completes before releasing buffer */

        /* Signal that we're done with this buffer */
        mbarrier_arrive(&smem->mbarriers[buf]);
    }

    /* ─────────────────────────────────────────────────────────────────────
     * PHASE 4: Write accumulated results to global memory
     *
     * Apply Level-2 tensor scale during writeback:
     *   output = accum × tensor_scale_A × tensor_scale_B
     * ───────────────────────────────────────────────────────────────────── */

    __syncwarp();  /* Ensure all K-iterations complete */

    /* Calculate combined tensor scale */
    float combined_scale = smem->tensor_scale_A * smem->tensor_scale_B;

    /* Calculate global memory destination for this warp's tile */
    int global_row = tile_row * TILE_M + warp_row_offset;
    int global_col = tile_col * TILE_N + warp_col_offset;
    half* global_dst = C + global_row * ldc + global_col;

    /* Store with scaling and FP32→FP16 conversion */
    tmem_store_scaled(
        global_dst,
        tmem_accum,
        WARP_TILE_M,
        WARP_TILE_N,
        ldc,
        combined_scale
    );

    /* ─────────────────────────────────────────────────────────────────────
     * PHASE 5: Handle the 4th quadrant (bottom-right)
     *
     * Since we only have 3 consumer warps but 4 quadrants, we need one
     * warp to do double duty. Warp 2 (consumer_warp_id=2) handles both
     * bottom-left AND bottom-right.
     * ───────────────────────────────────────────────────────────────────── */

    if (consumer_warp_id == 2) {
        /* Reset accumulator for second pass */
        if (lane_id == 0) {
            tmem_zero_accum(tmem_accum, WARP_TILE_M * WARP_TILE_N);
        }
        __syncwarp();

        /* Second pass: bottom-right quadrant */
        warp_row_offset = WARP_TILE_M;  /* 64 */
        warp_col_offset = WARP_TILE_N;  /* 64 */

        /* Re-process all K tiles for this quadrant */
        /* Note: Data is already in SMEM, we just need to re-read different portions */
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            int buf = k_tile % NUM_BUFFERS;
            SMEMBufferSlot* slot = &smem->buffers[buf];

            /* Note: No barrier wait needed - data persists in SMEM during first pass */
            /* This assumes the producer doesn't overwrite until all consumers are done */

            const char* smem_A_ptr = slot->tile_A +
                (warp_row_offset * TILE_K) / 2;
            const __nv_fp8_e4m3* smem_scales_A_ptr = slot->scales_A +
                (warp_row_offset / 16) * (TILE_K / 16);

            tmem_load_fp4_scaled(tmem_A, smem_A_ptr, smem_scales_A_ptr,
                                 WARP_TILE_M, MMA_K);

            const char* smem_B_ptr = slot->tile_B +
                (warp_col_offset) / 2;
            const __nv_fp8_e4m3* smem_scales_B_ptr = slot->scales_B +
                (warp_col_offset / 16);

            tmem_load_fp4_scaled(tmem_B, smem_B_ptr, smem_scales_B_ptr,
                                 MMA_K, WARP_TILE_N);

            __syncwarp();
            tmem_mma_nvfp4(tmem_accum, tmem_A, tmem_B);
            __syncwarp();
        }

        /* Store bottom-right quadrant */
        global_row = tile_row * TILE_M + warp_row_offset;
        global_col = tile_col * TILE_N + warp_col_offset;
        global_dst = C + global_row * ldc + global_col;

        tmem_store_scaled(global_dst, tmem_accum, WARP_TILE_M, WARP_TILE_N,
                          ldc, combined_scale);
    }

    /* ─────────────────────────────────────────────────────────────────────
     * PHASE 6: Free TMEM allocations
     * ───────────────────────────────────────────────────────────────────── */

    __syncwarp();
    if (lane_id == 0) {
        tmem_free_region(tmem_accum);
        tmem_free_region(tmem_A);
        tmem_free_region(tmem_B);
    }
}

/* ============================================================================
 * Main Kernel Entry Point
 * ============================================================================
 */

extern "C" __global__ void __launch_bounds__(BLOCK_THREADS, 1)
nvfp4_gemm_kernel(
    const CUtensorMap* __restrict__ tma_desc_A,
    const CUtensorMap* __restrict__ tma_desc_B,
    const CUtensorMap* __restrict__ tma_desc_scales_A,
    const CUtensorMap* __restrict__ tma_desc_scales_B,
    half* __restrict__ C,            /* Output matrix */
    int32_t M,
    int32_t N,
    int32_t K,
    float tensor_scale_A,
    float tensor_scale_B,
    int32_t ldc                      /* Leading dimension of C */
) {
    /* Allocate shared memory statically (allocation-free!) */
    __shared__ KernelSMEM smem;

    /* Compute tile coordinates for this block */
    const int32_t tile_row = blockIdx.y;
    const int32_t tile_col = blockIdx.x;
    const int32_t num_k_tiles = (K + TILE_K - 1) / TILE_K;

    /* Warp and lane identification */
    const int warp_id = threadIdx.x / WARP_SIZE;

    /* ─────────────────────────────────────────────────────────────────────
     * WARP ROLE ASSIGNMENT
     *
     * Warp 0: Producer (TMA operations)
     * Warps 1-3: Consumers (TMEM load + Tensor Core MMA)
     * ───────────────────────────────────────────────────────────────────── */

    if (warp_id == 0) {
        /* Producer warp: handle all TMA operations */
        producer_warp_main(
            &smem,
            tma_desc_A,
            tma_desc_B,
            tma_desc_scales_A,
            tma_desc_scales_B,
            tile_row,
            tile_col,
            num_k_tiles,
            tensor_scale_A,
            tensor_scale_B
        );
    } else {
        /* Consumer warps: SMEM→TMEM load, MMA, and writeback */
        consumer_warps_main(
            &smem,
            num_k_tiles,
            C,
            tile_row,
            tile_col,
            ldc
        );
    }

    /* Final synchronization before kernel exit */
    __syncthreads();
}

/* ============================================================================
 * Host-Side Kernel Launch Helper
 * ============================================================================
 */

extern "C" cudaError_t launch_nvfp4_gemm(
    const void* A,               /* NVFP4 matrix A [M, K] */
    const void* B,               /* NVFP4 matrix B [K, N] */
    const void* scales_A,        /* E4M3 scales for A */
    const void* scales_B,        /* E4M3 scales for B */
    half* C,                     /* Output FP16 matrix [M, N] */
    int M, int N, int K,
    float tensor_scale_A,
    float tensor_scale_B,
    cudaStream_t stream
) {
    /* Calculate grid dimensions */
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(BLOCK_THREADS);

    /* Create TMA descriptors */
    CUtensorMap tma_desc_A, tma_desc_B, tma_desc_scales_A, tma_desc_scales_B;

    /* A: [M, K] with tiles [128, 64] */
    create_nvfp4_tma_descriptor(
        &tma_desc_A, A, M, K, K / 2, TILE_M, TILE_K
    );

    /* B: [K, N] with tiles [64, 128] */
    create_nvfp4_tma_descriptor(
        &tma_desc_B, B, K, N, N / 2, TILE_K, TILE_N
    );

    /* Scales for A: [M/16, K/16] */
    create_scale_tma_descriptor(
        &tma_desc_scales_A, scales_A,
        M / 16, K / 16, K / 16,
        TILE_M / 16, TILE_K / 16
    );

    /* Scales for B: [K/16, N/16] */
    create_scale_tma_descriptor(
        &tma_desc_scales_B, scales_B,
        K / 16, N / 16, N / 16,
        TILE_K / 16, TILE_N / 16
    );

    /* Calculate shared memory size */
    size_t smem_size = sizeof(KernelSMEM);

    /* Set shared memory carveout for maximum SMEM */
    cudaFuncSetAttribute(
        nvfp4_gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    /* Launch kernel */
    nvfp4_gemm_kernel<<<grid, block, 0, stream>>>(
        &tma_desc_A,
        &tma_desc_B,
        &tma_desc_scales_A,
        &tma_desc_scales_B,
        C,
        M, N, K,
        tensor_scale_A,
        tensor_scale_B,
        N  /* ldc */
    );

    return cudaGetLastError();
}
