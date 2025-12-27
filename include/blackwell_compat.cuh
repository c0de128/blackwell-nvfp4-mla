/*
 * Blackwell Compatibility Header for NVFP4 Challenge
 *
 * This header provides a compatibility layer that allows development on Ampere
 * (sm_86) while targeting Blackwell (sm_100/sm_90a) for deployment.
 *
 * Two-Level Micro-Block Scaling Strategy:
 * ========================================
 * NVFP4 (E2M1) has only 4 bits per element, giving a very limited dynamic range.
 * To maintain numerical precision across GEMM/Attention operations, we use a
 * two-level scaling approach:
 *
 * Level 1 - Micro-Block Scale (E4M3):
 *   - Every 16 FP4 elements share one FP8 E4M3 scale factor
 *   - This is the "1x16 micro-block" pattern from Blackwell's MX format
 *   - The E4M3 scale provides 4 exponent bits for local dynamic range
 *   - Stored alongside the FP4 data for efficient memory access
 *
 * Level 2 - Tensor-Level Scale (FP32):
 *   - Each tensor (or tile in GEMM parlance) has one FP32 scale factor
 *   - This captures the global magnitude of the data
 *   - Applied during accumulation to maintain precision
 *   - Stored separately, typically one per 128x128 or 256x128 tile
 *
 * Dequantization formula:
 *   float_value = fp4_value * micro_block_scale_e4m3 * tensor_scale_fp32
 *
 * This two-level approach allows FP4's 4-bit precision to effectively span
 * a much larger dynamic range, critical for attention score computation
 * where values can vary significantly.
 */

#ifndef __BLACKWELL_COMPAT_CUH__
#define __BLACKWELL_COMPAT_CUH__

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

/* Architecture detection macros */
#define BLACKWELL_ARCH_THRESHOLD 1000

#if defined(__CUDA_ARCH__)
    #if __CUDA_ARCH__ >= BLACKWELL_ARCH_THRESHOLD
        #define IS_BLACKWELL 1
        #define IS_AMPERE 0
    #else
        #define IS_BLACKWELL 0
        #define IS_AMPERE 1
    #endif
#else
    /* Host code - default to Ampere compatibility mode */
    #define IS_BLACKWELL 0
    #define IS_AMPERE 1
#endif

/* Alignment requirements for optimal memory access */
#define NVFP4_BLOCK_ALIGN 16    /* 128-bit alignment for coalesced access */
#define NVFP4_ELEMENTS_PER_BLOCK 16

/*
 * NVFP4Block: Core data structure for micro-block scaled FP4 data
 *
 * Memory Layout (9 bytes logical, 16 bytes aligned):
 *   [0-7]:   8 bytes = 16 x 4-bit E2M1 values (packed)
 *   [8]:     1 byte  = E4M3 scale factor for this block
 *   [9-15]:  padding for alignment
 *
 * The 16 FP4 values are packed into 8 bytes (2 values per byte).
 * Lower nibble = even index, upper nibble = odd index.
 */
struct __align__(NVFP4_BLOCK_ALIGN) NVFP4Block {
    /* 16 FP4 E2M1 values packed into 4 x fp4x4 (8 bytes total) */
    __nv_fp4x4_e2m1 data[4];

    /* Per-block scale factor in E4M3 format */
    __nv_fp8_e4m3 scale;

    /* Padding to maintain 16-byte alignment */
    unsigned char _pad[7];

    /* Host/Device constructor */
    __host__ __device__ NVFP4Block() : scale() {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            data[i].__x = 0;
        }
    }
};

/*
 * NVFP4Tensor: Container for a tile of FP4 data with two-level scaling
 *
 * This represents a logical tile (e.g., 128x128) composed of many NVFP4Blocks,
 * plus a single FP32 tensor-level scale factor.
 */
struct NVFP4TensorScale {
    float tensor_scale;      /* Level 2: Tensor-wide FP32 scale */
    float tensor_scale_inv;  /* Precomputed 1.0f / tensor_scale for efficiency */

    __host__ __device__ NVFP4TensorScale(float s = 1.0f)
        : tensor_scale(s), tensor_scale_inv(1.0f / s) {}
};

/* ============================================================================
 * TMEM (Tensor Memory) Compatibility Layer
 *
 * Blackwell introduces Tensor Memory (TMEM) - a new memory hierarchy level
 * between registers and shared memory. TMEM provides:
 *   - Higher capacity than registers (reduces register pressure)
 *   - Lower latency than shared memory
 *   - Direct feed to Tensor Cores via tcgen05 instructions
 *
 * On Ampere (sm_86), we mock these operations using shared memory.
 * ============================================================================
 */

#if IS_BLACKWELL

/* Blackwell: Use actual TMEM instructions via PTX inline assembly */

__device__ __forceinline__ void tmem_alloc(void** ptr, size_t size) {
    /* tcgen05.alloc - allocate tensor memory
     * This is a placeholder for the actual PTX instruction.
     * Real implementation requires specific PTX syntax for sm_100+ */
    asm volatile(
        "// TMEM allocation placeholder for Blackwell\n"
        "// tcgen05.alloc %0, %1;\n"
        : "=l"(*ptr)
        : "l"(size)
    );
}

__device__ __forceinline__ void tmem_free(void* ptr) {
    /* tcgen05.free - release tensor memory */
    asm volatile(
        "// TMEM free placeholder for Blackwell\n"
        "// tcgen05.free %0;\n"
        :
        : "l"(ptr)
    );
}

__device__ __forceinline__ void tmem_load_async(void* dst, const void* src, size_t size) {
    /* Asynchronous load from global/shared to TMEM */
    asm volatile(
        "// TMEM async load placeholder\n"
        "// tcgen05.ld.async %0, %1, %2;\n"
        :
        : "l"(dst), "l"(src), "l"(size)
        : "memory"
    );
}

#else /* IS_AMPERE or Host */

/* Ampere: Mock TMEM operations using shared memory */

__device__ __forceinline__ void tmem_alloc_mock(void** ptr, size_t size) {
    /* On Ampere, TMEM doesn't exist. This is a no-op placeholder.
     * Actual shared memory allocation should be done statically. */
    *ptr = nullptr;
}

__device__ __forceinline__ void tmem_free_mock(void* ptr) {
    /* No-op on Ampere */
    (void)ptr;
}

__device__ __forceinline__ void tmem_load_async_mock(void* dst, const void* src, size_t size) {
    /* On Ampere, fall back to synchronous copy or cp.async for shared memory */
    (void)dst;
    (void)src;
    (void)size;
}

/* Alias mock functions for uniform API */
#define tmem_alloc(ptr, size) tmem_alloc_mock(ptr, size)
#define tmem_free(ptr) tmem_free_mock(ptr)
#define tmem_load_async(dst, src, size) tmem_load_async_mock(dst, src, size)

#endif /* IS_BLACKWELL */

/* ============================================================================
 * Asynchronous Copy Utilities (cp.async)
 *
 * Both Ampere and Blackwell support cp.async for efficient global->shared
 * memory transfers. These wrappers ensure consistent usage.
 * ============================================================================
 */

__device__ __forceinline__ void cp_async_128bit(void* dst, const void* src) {
    /* cp.async.cg.shared.global for 16-byte (128-bit) transfers */
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "l"(dst), "l"(src)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

/* Template version for compile-time constant group count */
template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    /* Wait for all but the most recent N groups (N must be compile-time constant) */
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

/* Common specializations for convenience */
__device__ __forceinline__ void cp_async_wait_group_0() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group_1() {
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}

/* ============================================================================
 * Stochastic Rounding RNG
 *
 * Lightweight xor-shift RNG for stochastic rounding during quantization.
 * Stochastic rounding probabilistically rounds up or down based on the
 * fractional distance to the nearest representable value, which provides
 * unbiased gradient flow during training and reduces quantization bias.
 *
 * Algorithm: xorshift32 (Marsaglia, 2003)
 * Period: 2^32 - 1
 * Speed: 3 XOR + 3 shift operations
 * ============================================================================
 */

struct StochasticRNG {
    uint32_t state;

    __device__ __forceinline__ StochasticRNG(uint32_t seed = 0) {
        /* Initialize with thread-unique seed */
        state = seed ^ (threadIdx.x + 1) ^ ((blockIdx.x + 1) << 16);
        if (state == 0) state = 0xDEADBEEF;  /* Ensure non-zero state */
    }

    __device__ __forceinline__ uint32_t next() {
        /* xorshift32 algorithm */
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    __device__ __forceinline__ float uniform() {
        /* Returns uniform float in [0, 1) */
        return (next() & 0x7FFFFF) * (1.0f / 8388608.0f);  /* 2^-23 */
    }
};

/*
 * Create a thread-local RNG with deterministic seeding
 * Call this at the start of your kernel to get a unique RNG per thread
 */
__device__ __forceinline__ StochasticRNG make_thread_rng(uint32_t extra_seed = 0) {
    uint32_t seed = threadIdx.x
                  ^ (threadIdx.y << 10)
                  ^ (threadIdx.z << 20)
                  ^ (blockIdx.x << 5)
                  ^ (blockIdx.y << 15)
                  ^ (blockIdx.z << 25)
                  ^ extra_seed
                  ^ 0xCAFEBABE;
    return StochasticRNG(seed);
}

/* ============================================================================
 * FP4 Conversion Utilities
 *
 * Helper functions for converting between FP4 and higher precision formats,
 * incorporating the two-level scaling.
 * ============================================================================
 */

/* E2M1 representable values (positive half) for stochastic rounding */
__device__ __constant__ float NVFP4_STOCHASTIC_VALUES[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

__device__ __forceinline__ float nvfp4_to_float(
    __nv_fp4_e2m1 val,
    __nv_fp8_e4m3 block_scale,
    float tensor_scale
) {
    /* Dequantize: fp4 -> fp32 with two-level scaling */
    float fp4_val = static_cast<float>(val);
    float scale_val = static_cast<float>(block_scale);
    return fp4_val * scale_val * tensor_scale;
}

__host__ __device__ __forceinline__ __nv_fp4_e2m1 float_to_nvfp4(
    float val,
    float block_scale_inv,
    float tensor_scale_inv
) {
    /* Quantize: fp32 -> fp4 with two-level scaling (truncation) */
    float scaled = val * tensor_scale_inv * block_scale_inv;
    return __nv_fp4_e2m1(scaled);
}

/*
 * Stochastic Rounding Quantization
 *
 * Instead of truncating to the nearest representable FP4 value, we
 * probabilistically round up or down based on the fractional distance.
 *
 * For value x between representable values a and b (a < x < b):
 *   P(round to b) = (x - a) / (b - a)
 *   P(round to a) = (b - x) / (b - a)
 *
 * This ensures E[quantized(x)] = x (unbiased quantization).
 *
 * Benefits:
 *   - Reduces systematic quantization bias
 *   - Improves gradient flow in training
 *   - Better preserves statistical properties of tensors
 */
__device__ __forceinline__ __nv_fp4_e2m1 float_to_nvfp4_stochastic(
    float val,
    float block_scale_inv,
    float tensor_scale_inv,
    StochasticRNG& rng
) {
    /* Apply scales to get value in FP4 representable range */
    float scaled = val * tensor_scale_inv * block_scale_inv;

    /* Handle sign separately */
    bool negative = (scaled < 0.0f);
    float abs_scaled = negative ? -scaled : scaled;

    /* Clamp to FP4 range [0, 6] */
    abs_scaled = fminf(abs_scaled, 6.0f);
    abs_scaled = fmaxf(abs_scaled, 0.0f);

    /* Find bracketing FP4 values */
    int lower_idx = 0;
    for (int i = 7; i >= 0; i--) {
        if (NVFP4_STOCHASTIC_VALUES[i] <= abs_scaled) {
            lower_idx = i;
            break;
        }
    }
    int upper_idx = (lower_idx < 7) ? (lower_idx + 1) : 7;

    float lower_val = NVFP4_STOCHASTIC_VALUES[lower_idx];
    float upper_val = NVFP4_STOCHASTIC_VALUES[upper_idx];

    /* Compute stochastic rounding probability */
    float range = upper_val - lower_val;
    int final_idx;

    if (range < 1e-6f) {
        /* Value is exactly representable */
        final_idx = lower_idx;
    } else {
        /* Probabilistic rounding */
        float prob_up = (abs_scaled - lower_val) / range;
        float rand_val = rng.uniform();

        final_idx = (rand_val < prob_up) ? upper_idx : lower_idx;
    }

    /* Apply sign and convert to FP4 */
    if (negative && final_idx > 0) {
        final_idx |= 0x8;  /* Set sign bit for negative values */
    }

    /* Return as __nv_fp4_e2m1 */
    __nv_fp4_e2m1 result;
    /* Pack into the internal representation */
    float final_val = NVFP4_STOCHASTIC_VALUES[final_idx & 0x7];
    if (negative) final_val = -final_val;
    result = __nv_fp4_e2m1(final_val);

    return result;
}

/*
 * Batch stochastic quantization for a micro-block (16 elements)
 *
 * This is optimized for quantizing entire micro-blocks at once,
 * which is the common use case in GEMM/Attention kernels.
 */
__device__ __forceinline__ void quantize_block_stochastic(
    const float* input,         /* 16 FP32 values */
    __nv_fp4x4_e2m1* output,    /* 4 packed FP4x4 outputs */
    __nv_fp8_e4m3& scale_out,   /* Computed E4M3 scale */
    float tensor_scale_inv,
    StochasticRNG& rng
) {
    /* Find max absolute value for block scale */
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        max_abs = fmaxf(max_abs, fabsf(input[i]));
    }

    /* Compute block scale (map max value to FP4 max of 6.0) */
    float block_scale = max_abs / 6.0f;
    if (block_scale < 1e-10f) block_scale = 1.0f;  /* Avoid division by zero */
    float block_scale_inv = 1.0f / block_scale;

    /* Store scale as E4M3 */
    scale_out = __nv_fp8_e4m3(block_scale);

    /* Quantize each element with stochastic rounding */
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        __nv_fp4_e2m1 v0 = float_to_nvfp4_stochastic(input[i*4 + 0], block_scale_inv, tensor_scale_inv, rng);
        __nv_fp4_e2m1 v1 = float_to_nvfp4_stochastic(input[i*4 + 1], block_scale_inv, tensor_scale_inv, rng);
        __nv_fp4_e2m1 v2 = float_to_nvfp4_stochastic(input[i*4 + 2], block_scale_inv, tensor_scale_inv, rng);
        __nv_fp4_e2m1 v3 = float_to_nvfp4_stochastic(input[i*4 + 3], block_scale_inv, tensor_scale_inv, rng);

        /* Pack 4 FP4 values into fp4x4 */
        /* Note: This packing depends on __nv_fp4x4_e2m1 internal representation */
        output[i].__x = ((unsigned char)v0.__x) |
                        ((unsigned char)v1.__x << 4) |
                        ((unsigned char)v2.__x << 8) |
                        ((unsigned char)v3.__x << 12);
    }
}

/* ============================================================================
 * Memory Alignment Helpers
 * ============================================================================
 */

template<typename T>
__host__ __device__ __forceinline__ bool is_aligned_128bit(const T* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

#endif /* __BLACKWELL_COMPAT_CUH__ */
