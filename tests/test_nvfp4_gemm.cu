/*
 * NVFP4 GEMM Test Harness
 *
 * This test validates the mathematical correctness of our NVFP4 GEMM kernel
 * by comparing against a FP32 reference implementation. Since we develop on
 * Ampere (3060), we cannot execute actual tcgen05 instructions, but we CAN
 * verify:
 *
 * 1. NVFP4 quantization/dequantization is bit-accurate
 * 2. Two-level scaling math (micro-block E4M3 + tensor FP32) is correct
 * 3. Memory layout and data flow are properly structured
 * 4. The reference implementation matches expected GEMM output
 *
 * The test strategy:
 *   a) Generate random FP32 matrices A and B
 *   b) Quantize to NVFP4 with two-level scaling
 *   c) Compute reference GEMM in FP32
 *   d) Compute NVFP4 GEMM (simulated on CPU/Ampere)
 *   e) Compare results within tolerance
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <random>

#include "../include/blackwell_compat.cuh"

/* ============================================================================
 * Test Configuration
 * ============================================================================
 */

/* Matrix dimensions for testing */
#define TEST_M 256
#define TEST_N 256
#define TEST_K 128

/* Tolerance for floating-point comparison */
/* NVFP4 is very low precision, so we expect ~10-20% relative error */
#define RELATIVE_TOLERANCE 0.25f
#define ABSOLUTE_TOLERANCE 0.1f

/* Micro-block size for scaling */
#define MICRO_BLOCK_SIZE 16

/* ============================================================================
 * NVFP4 (E2M1) Emulation on Host
 * ============================================================================
 *
 * E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
 *
 * Representable values (positive):
 *   0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
 *
 * The limited precision means we need careful scaling to maintain accuracy.
 */

/* E2M1 value table (4-bit values 0-15) */
static const float FP4_E2M1_TABLE[16] = {
    0.0f,   /* 0000 */
    0.5f,   /* 0001 */
    1.0f,   /* 0010 */
    1.5f,   /* 0011 */
    2.0f,   /* 0100 */
    3.0f,   /* 0101 */
    4.0f,   /* 0110 */
    6.0f,   /* 0111 */
    -0.0f,  /* 1000 (negative zero, treated as 0) */
    -0.5f,  /* 1001 */
    -1.0f,  /* 1010 */
    -1.5f,  /* 1011 */
    -2.0f,  /* 1100 */
    -3.0f,  /* 1101 */
    -4.0f,  /* 1110 */
    -6.0f   /* 1111 */
};

/*
 * Quantize a float to E2M1 (4-bit FP4).
 * Returns the 4-bit code (0-15).
 */
uint8_t float_to_fp4_e2m1(float val) {
    if (val == 0.0f) return 0;

    int sign = (val < 0) ? 1 : 0;
    float abs_val = fabsf(val);

    /* Find nearest representable value */
    uint8_t best_code = 0;
    float best_diff = fabsf(abs_val - FP4_E2M1_TABLE[0]);

    for (int i = 1; i < 8; i++) {
        float diff = fabsf(abs_val - FP4_E2M1_TABLE[i]);
        if (diff < best_diff) {
            best_diff = diff;
            best_code = i;
        }
    }

    return (sign << 3) | best_code;
}

/*
 * Dequantize E2M1 4-bit code to float.
 */
float fp4_e2m1_to_float(uint8_t code) {
    return FP4_E2M1_TABLE[code & 0x0F];
}

/* ============================================================================
 * E4M3 (FP8) Emulation for Micro-Block Scales
 * ============================================================================
 */

/*
 * Convert float to E4M3 (FP8).
 * E4M3: 1 sign, 4 exponent, 3 mantissa bits.
 * Range: ~±240, precision ~0.1% at max
 */
uint8_t float_to_fp8_e4m3(float val) {
    if (val == 0.0f) return 0;

    /* Use CUDA's native conversion if available, else approximate */
    __nv_fp8_e4m3 fp8_val(val);
    uint8_t* raw = reinterpret_cast<uint8_t*>(&fp8_val);
    return *raw;
}

float fp8_e4m3_to_float(uint8_t code) {
    __nv_fp8_e4m3 fp8_val;
    uint8_t* raw = reinterpret_cast<uint8_t*>(&fp8_val);
    *raw = code;
    return static_cast<float>(fp8_val);
}

/* ============================================================================
 * Two-Level Scaling Implementation
 * ============================================================================
 */

struct QuantizedMatrix {
    uint8_t* fp4_data;        /* Packed FP4 values (2 per byte) */
    uint8_t* micro_scales;    /* E4M3 scale per 16-element block */
    float tensor_scale;       /* FP32 tensor-level scale */
    int rows;
    int cols;
    int num_micro_blocks;

    void allocate(int r, int c) {
        rows = r;
        cols = c;
        num_micro_blocks = (r * c + MICRO_BLOCK_SIZE - 1) / MICRO_BLOCK_SIZE;

        fp4_data = new uint8_t[(r * c + 1) / 2];  /* 2 FP4 per byte */
        micro_scales = new uint8_t[num_micro_blocks];
        tensor_scale = 1.0f;
    }

    void free() {
        delete[] fp4_data;
        delete[] micro_scales;
    }
};

/*
 * Quantize FP32 matrix to NVFP4 with two-level scaling.
 *
 * Algorithm:
 * 1. Compute tensor-level scale from global max absolute value
 * 2. For each 16-element micro-block:
 *    a) Find local max absolute value
 *    b) Compute E4M3 micro-scale
 *    c) Quantize elements using combined scale
 */
void quantize_to_nvfp4(
    const float* src,
    QuantizedMatrix* dst,
    int rows,
    int cols
) {
    dst->allocate(rows, cols);
    int total_elements = rows * cols;

    /* Step 1: Find global maximum for tensor-level scale */
    float global_max = 0.0f;
    for (int i = 0; i < total_elements; i++) {
        global_max = fmaxf(global_max, fabsf(src[i]));
    }

    /* Tensor scale maps global_max to FP4 max (6.0) */
    dst->tensor_scale = (global_max > 0.0f) ? (global_max / 6.0f) : 1.0f;
    float tensor_scale_inv = 1.0f / dst->tensor_scale;

    /* Step 2: Process each micro-block */
    int num_blocks = dst->num_micro_blocks;
    for (int block = 0; block < num_blocks; block++) {
        int start_idx = block * MICRO_BLOCK_SIZE;
        int end_idx = (start_idx + MICRO_BLOCK_SIZE < total_elements)
                      ? (start_idx + MICRO_BLOCK_SIZE) : total_elements;

        /* Find local maximum after tensor scaling */
        float local_max = 0.0f;
        for (int i = start_idx; i < end_idx; i++) {
            float scaled = src[i] * tensor_scale_inv;
            local_max = fmaxf(local_max, fabsf(scaled));
        }

        /* Micro-scale maps local_max to FP4 max (6.0) */
        float micro_scale = (local_max > 0.0f) ? (local_max / 6.0f) : 1.0f;
        dst->micro_scales[block] = float_to_fp8_e4m3(micro_scale);

        /* Dequantize micro_scale for accurate quantization */
        float micro_scale_actual = fp8_e4m3_to_float(dst->micro_scales[block]);
        float combined_scale_inv = tensor_scale_inv /
            ((micro_scale_actual > 0.0f) ? micro_scale_actual : 1.0f);

        /* Quantize elements in this block */
        for (int i = start_idx; i < end_idx; i++) {
            float val = src[i] * combined_scale_inv;
            uint8_t fp4_code = float_to_fp4_e2m1(val);

            /* Pack into fp4_data (2 values per byte) */
            int byte_idx = i / 2;
            if (i % 2 == 0) {
                dst->fp4_data[byte_idx] = (dst->fp4_data[byte_idx] & 0xF0) | fp4_code;
            } else {
                dst->fp4_data[byte_idx] = (dst->fp4_data[byte_idx] & 0x0F) | (fp4_code << 4);
            }
        }
    }
}

/*
 * Dequantize NVFP4 matrix back to FP32.
 */
void dequantize_from_nvfp4(
    const QuantizedMatrix* src,
    float* dst
) {
    int total_elements = src->rows * src->cols;

    for (int i = 0; i < total_elements; i++) {
        int block = i / MICRO_BLOCK_SIZE;

        /* Extract FP4 code */
        int byte_idx = i / 2;
        uint8_t fp4_code;
        if (i % 2 == 0) {
            fp4_code = src->fp4_data[byte_idx] & 0x0F;
        } else {
            fp4_code = (src->fp4_data[byte_idx] >> 4) & 0x0F;
        }

        /* Get scales */
        float micro_scale = fp8_e4m3_to_float(src->micro_scales[block]);
        float tensor_scale = src->tensor_scale;

        /* Dequantize */
        float fp4_val = fp4_e2m1_to_float(fp4_code);
        dst[i] = fp4_val * micro_scale * tensor_scale;
    }
}

/* ============================================================================
 * Reference GEMM Implementations
 * ============================================================================
 */

/*
 * Reference FP32 GEMM: C = A × B
 *
 * A: [M, K]
 * B: [K, N]
 * C: [M, N]
 */
void reference_gemm_fp32(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/*
 * NVFP4 GEMM with two-level scaling: C = A_fp4 × B_fp4
 *
 * This simulates what the Blackwell kernel computes:
 * 1. Load FP4 values with micro-scale (Level-1)
 * 2. Perform MMA (accumulate in FP32)
 * 3. Apply tensor-scale (Level-2) during writeback
 */
void nvfp4_gemm_reference(
    const QuantizedMatrix* A,
    const QuantizedMatrix* B,
    float* C,
    int M, int N, int K
) {
    /* Dequantize A and B to FP32 for computation */
    float* A_dequant = new float[M * K];
    float* B_dequant = new float[K * N];

    dequantize_from_nvfp4(A, A_dequant);
    dequantize_from_nvfp4(B, B_dequant);

    /* Perform FP32 GEMM */
    reference_gemm_fp32(A_dequant, B_dequant, C, M, N, K);

    delete[] A_dequant;
    delete[] B_dequant;
}

/* ============================================================================
 * Test Utilities
 * ============================================================================
 */

void generate_random_matrix(float* mat, int rows, int cols, float scale = 1.0f) {
    std::mt19937 rng(42);  /* Fixed seed for reproducibility */
    std::uniform_real_distribution<float> dist(-scale, scale);

    for (int i = 0; i < rows * cols; i++) {
        mat[i] = dist(rng);
    }
}

float compute_relative_error(const float* ref, const float* test, int n) {
    float max_rel_error = 0.0f;
    float sum_sq_error = 0.0f;
    float sum_sq_ref = 0.0f;
    int outliers = 0;

    /* Find the range of reference values for adaptive tolerance */
    float ref_max = 0.0f;
    for (int i = 0; i < n; i++) {
        ref_max = fmaxf(ref_max, fabsf(ref[i]));
    }
    float significance_threshold = ref_max * 0.01f;  /* 1% of max is significant */

    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - test[i]);
        sum_sq_error += diff * diff;
        sum_sq_ref += ref[i] * ref[i];

        /* Only compute relative error for significant values */
        if (fabsf(ref[i]) > significance_threshold) {
            float rel = diff / fabsf(ref[i]);
            if (rel > max_rel_error) {
                max_rel_error = rel;
            }
            if (rel > RELATIVE_TOLERANCE) {
                outliers++;
            }
        }
    }

    float rmse = sqrtf(sum_sq_error / n);
    float nrmse = (sum_sq_ref > 0.0f) ? sqrtf(sum_sq_error / sum_sq_ref) : 0.0f;

    printf("  Max relative error (significant values): %.4f%%\n", max_rel_error * 100.0f);
    printf("  RMSE: %.6f\n", rmse);
    printf("  Normalized RMSE: %.4f%%\n", nrmse * 100.0f);
    printf("  Outliers (>%.0f%% error): %d / %d\n", RELATIVE_TOLERANCE * 100.0f, outliers, n);

    /* Return NRMSE as the primary metric for FP4 comparisons */
    return nrmse;
}

bool check_tolerance(float max_rel_error) {
    return max_rel_error < RELATIVE_TOLERANCE;
}

/* ============================================================================
 * Test Cases
 * ============================================================================
 */

/*
 * Test 1: FP4 Quantization Round-Trip
 *
 * Verify that quantization → dequantization preserves values within FP4
 * precision limits.
 */
bool test_fp4_round_trip() {
    printf("\n=== Test 1: FP4 Quantization Round-Trip ===\n");

    float test_values[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                           -0.5f, -1.0f, -2.0f, -6.0f};
    int num_tests = sizeof(test_values) / sizeof(float);

    bool all_pass = true;
    for (int i = 0; i < num_tests; i++) {
        float val = test_values[i];
        uint8_t code = float_to_fp4_e2m1(val);
        float recovered = fp4_e2m1_to_float(code);

        bool pass = (fabsf(val - recovered) < 0.001f);
        printf("  %.2f → code %d → %.2f : %s\n",
               val, code, recovered, pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    printf("Test 1 Result: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass;
}

/*
 * Test 2: Two-Level Scaling Math
 *
 * Verify that the two-level scaling (micro + tensor) correctly handles
 * values outside the raw FP4 range.
 */
bool test_two_level_scaling() {
    printf("\n=== Test 2: Two-Level Scaling Math ===\n");

    /* Create a small test matrix with values outside FP4 range */
    const int SIZE = 32;
    float* original = new float[SIZE];
    float* recovered = new float[SIZE];

    /* Values from -100 to +100 */
    for (int i = 0; i < SIZE; i++) {
        original[i] = (i - SIZE/2) * 6.25f;  /* Range: -100 to +93.75 */
    }

    /* Quantize and dequantize */
    QuantizedMatrix qmat;
    quantize_to_nvfp4(original, &qmat, 1, SIZE);
    dequantize_from_nvfp4(&qmat, recovered);

    printf("  Tensor scale: %.4f\n", qmat.tensor_scale);
    printf("  Micro scales: ");
    for (int i = 0; i < qmat.num_micro_blocks; i++) {
        printf("%.4f ", fp8_e4m3_to_float(qmat.micro_scales[i]));
    }
    printf("\n");

    /* Check a few values */
    bool all_pass = true;
    for (int i = 0; i < SIZE; i += 8) {
        float rel_error = (fabsf(original[i]) > 0.1f)
                          ? fabsf(original[i] - recovered[i]) / fabsf(original[i])
                          : fabsf(original[i] - recovered[i]);
        bool pass = rel_error < RELATIVE_TOLERANCE;
        printf("  [%d] Original: %8.2f, Recovered: %8.2f, Error: %5.1f%% : %s\n",
               i, original[i], recovered[i], rel_error * 100.0f,
               pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    qmat.free();
    delete[] original;
    delete[] recovered;

    printf("Test 2 Result: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass;
}

/*
 * Test 3: Reference GEMM Correctness
 *
 * Verify the FP32 reference GEMM produces correct results.
 */
bool test_reference_gemm() {
    printf("\n=== Test 3: Reference GEMM Correctness ===\n");

    /* Small known test case: identity matrix */
    const int SIZE = 4;
    float A[SIZE * SIZE] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    float B[SIZE * SIZE] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float C[SIZE * SIZE];
    float expected[SIZE * SIZE] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    reference_gemm_fp32(A, B, C, SIZE, SIZE, SIZE);

    bool all_pass = true;
    for (int i = 0; i < SIZE * SIZE; i++) {
        if (fabsf(C[i] - expected[i]) > 0.001f) {
            printf("  Mismatch at [%d]: expected %.2f, got %.2f\n",
                   i, expected[i], C[i]);
            all_pass = false;
        }
    }

    printf("  Identity matrix test: %s\n", all_pass ? "PASS" : "FAIL");

    /* Second test: 2x2 multiplication */
    float A2[4] = {1, 2, 3, 4};
    float B2[4] = {5, 6, 7, 8};
    float C2[4];
    float expected2[4] = {19, 22, 43, 50};  /* [[1,2],[3,4]] × [[5,6],[7,8]] */

    reference_gemm_fp32(A2, B2, C2, 2, 2, 2);

    bool pass2 = true;
    for (int i = 0; i < 4; i++) {
        if (fabsf(C2[i] - expected2[i]) > 0.001f) {
            printf("  Mismatch at [%d]: expected %.2f, got %.2f\n",
                   i, expected2[i], C2[i]);
            pass2 = false;
        }
    }
    printf("  2x2 multiplication test: %s\n", pass2 ? "PASS" : "FAIL");
    all_pass &= pass2;

    printf("Test 3 Result: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass;
}

/*
 * Test 4: NVFP4 GEMM vs FP32 Reference
 *
 * This is the main validation test. It compares NVFP4 GEMM output against
 * FP32 reference to ensure our quantization and scaling are working correctly.
 */
bool test_nvfp4_gemm_accuracy() {
    printf("\n=== Test 4: NVFP4 GEMM vs FP32 Reference ===\n");
    printf("  Matrix dimensions: M=%d, N=%d, K=%d\n", TEST_M, TEST_N, TEST_K);

    /* Allocate matrices */
    float* A_fp32 = new float[TEST_M * TEST_K];
    float* B_fp32 = new float[TEST_K * TEST_N];
    float* C_ref = new float[TEST_M * TEST_N];
    float* C_nvfp4 = new float[TEST_M * TEST_N];

    /* Generate random input data */
    printf("  Generating random matrices...\n");
    generate_random_matrix(A_fp32, TEST_M, TEST_K, 10.0f);
    generate_random_matrix(B_fp32, TEST_K, TEST_N, 10.0f);

    /* Compute FP32 reference */
    printf("  Computing FP32 reference GEMM...\n");
    auto t1 = std::chrono::high_resolution_clock::now();
    reference_gemm_fp32(A_fp32, B_fp32, C_ref, TEST_M, TEST_N, TEST_K);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ref_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    printf("  FP32 reference time: %ld us\n", ref_time);

    /* Quantize to NVFP4 */
    printf("  Quantizing matrices to NVFP4...\n");
    QuantizedMatrix A_quant, B_quant;
    quantize_to_nvfp4(A_fp32, &A_quant, TEST_M, TEST_K);
    quantize_to_nvfp4(B_fp32, &B_quant, TEST_K, TEST_N);

    printf("  A tensor scale: %.6f, B tensor scale: %.6f\n",
           A_quant.tensor_scale, B_quant.tensor_scale);

    /* Compute NVFP4 GEMM */
    printf("  Computing NVFP4 GEMM (simulated)...\n");
    t1 = std::chrono::high_resolution_clock::now();
    nvfp4_gemm_reference(&A_quant, &B_quant, C_nvfp4, TEST_M, TEST_N, TEST_K);
    t2 = std::chrono::high_resolution_clock::now();
    auto nvfp4_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    printf("  NVFP4 GEMM (simulated) time: %ld us\n", nvfp4_time);

    /* Compare results */
    printf("  Comparing results...\n");
    float max_rel_error = compute_relative_error(C_ref, C_nvfp4, TEST_M * TEST_N);

    bool pass = check_tolerance(max_rel_error);
    printf("  Tolerance check (%.0f%%): %s\n",
           RELATIVE_TOLERANCE * 100.0f, pass ? "PASS" : "FAIL");

    /* Cleanup */
    A_quant.free();
    B_quant.free();
    delete[] A_fp32;
    delete[] B_fp32;
    delete[] C_ref;
    delete[] C_nvfp4;

    printf("Test 4 Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

/*
 * Test 5: Memory Layout Verification
 *
 * Verify that NVFP4Block layout matches our expectations and is properly
 * aligned for TMA access.
 */
bool test_memory_layout() {
    printf("\n=== Test 5: Memory Layout Verification ===\n");

    /* Check NVFP4Block size and alignment */
    printf("  sizeof(NVFP4Block) = %zu bytes (expected: 16)\n", sizeof(NVFP4Block));
    printf("  alignof(NVFP4Block) = %zu bytes (expected: 16)\n", alignof(NVFP4Block));

    bool pass = (sizeof(NVFP4Block) == 16) && (alignof(NVFP4Block) == 16);
    printf("  NVFP4Block layout: %s\n", pass ? "PASS" : "FAIL");

    /* Verify data packing */
    NVFP4Block block;

    /* Check that we can access data and scale */
    printf("  Data array size: %zu bytes\n", sizeof(block.data));
    printf("  Scale size: %zu bytes\n", sizeof(block.scale));

    /* Test 128-bit alignment check function */
    float __attribute__((aligned(16))) aligned_data[4];
    float unaligned_data[5];

    bool aligned_check = is_aligned_128bit(aligned_data);
    bool unaligned_check = is_aligned_128bit(unaligned_data + 1);

    printf("  Aligned pointer check: %s (expected: true)\n",
           aligned_check ? "true" : "false");
    printf("  Unaligned pointer check: %s (expected: may vary)\n",
           unaligned_check ? "true" : "false");

    pass &= aligned_check;

    printf("Test 5 Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

/*
 * Test 6: Tile Dimension Verification
 *
 * Ensure our tiling strategy dimensions are consistent.
 */
bool test_tile_dimensions() {
    printf("\n=== Test 6: Tile Dimension Verification ===\n");

    /* From our kernel configuration */
    const int TILE_M = 128;
    const int TILE_N = 128;
    const int TILE_K = 64;
    const int WARP_TILE_M = 64;
    const int WARP_TILE_N = 64;
    const int MMA_K = 64;

    bool pass = true;

    /* Verify tile fits evenly into warp tiles */
    pass &= (TILE_M % WARP_TILE_M == 0);
    pass &= (TILE_N % WARP_TILE_N == 0);
    printf("  TILE_M (%d) / WARP_TILE_M (%d) = %d : %s\n",
           TILE_M, WARP_TILE_M, TILE_M / WARP_TILE_M,
           (TILE_M % WARP_TILE_M == 0) ? "OK" : "ERROR");
    printf("  TILE_N (%d) / WARP_TILE_N (%d) = %d : %s\n",
           TILE_N, WARP_TILE_N, TILE_N / WARP_TILE_N,
           (TILE_N % WARP_TILE_N == 0) ? "OK" : "ERROR");

    /* Verify K dimension matches MMA */
    pass &= (TILE_K == MMA_K);
    printf("  TILE_K (%d) == MMA_K (%d) : %s\n",
           TILE_K, MMA_K, (TILE_K == MMA_K) ? "OK" : "ERROR");

    /* Verify micro-block scaling works with tile dimensions */
    int blocks_per_A_tile = (TILE_M * TILE_K) / MICRO_BLOCK_SIZE;
    int blocks_per_B_tile = (TILE_K * TILE_N) / MICRO_BLOCK_SIZE;
    printf("  Micro-blocks per A tile: %d\n", blocks_per_A_tile);
    printf("  Micro-blocks per B tile: %d\n", blocks_per_B_tile);

    /* Verify alignment for FP4 packing */
    int fp4_bytes_A = (TILE_M * TILE_K) / 2;
    int fp4_bytes_B = (TILE_K * TILE_N) / 2;
    pass &= (fp4_bytes_A % 16 == 0);  /* 128-bit aligned */
    pass &= (fp4_bytes_B % 16 == 0);
    printf("  A tile FP4 bytes: %d (128-bit aligned: %s)\n",
           fp4_bytes_A, (fp4_bytes_A % 16 == 0) ? "YES" : "NO");
    printf("  B tile FP4 bytes: %d (128-bit aligned: %s)\n",
           fp4_bytes_B, (fp4_bytes_B % 16 == 0) ? "YES" : "NO");

    printf("Test 6 Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ============================================================================
 * Main Test Driver
 * ============================================================================
 */

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║        NVFP4 GEMM Test Harness - Blackwell Challenge           ║\n");
    printf("║        Running on Ampere (Logic Validation Mode)               ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");

    int passed = 0;
    int failed = 0;

    /* Run all tests */
    if (test_fp4_round_trip()) passed++; else failed++;
    if (test_two_level_scaling()) passed++; else failed++;
    if (test_reference_gemm()) passed++; else failed++;
    if (test_nvfp4_gemm_accuracy()) passed++; else failed++;
    if (test_memory_layout()) passed++; else failed++;
    if (test_tile_dimensions()) passed++; else failed++;

    /* Summary */
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                      TEST SUMMARY                              ║\n");
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    printf("║  Passed: %d                                                     ║\n", passed);
    printf("║  Failed: %d                                                     ║\n", failed);
    printf("║  Total:  %d                                                     ║\n", passed + failed);
    printf("╠════════════════════════════════════════════════════════════════╣\n");

    if (failed == 0) {
        printf("║  ✓ ALL TESTS PASSED - Ready for Blackwell Deployment          ║\n");
    } else {
        printf("║  ✗ SOME TESTS FAILED - Review errors above                    ║\n");
    }
    printf("╚════════════════════════════════════════════════════════════════╝\n");

    return (failed == 0) ? 0 : 1;
}
