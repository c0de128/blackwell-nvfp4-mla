/* Test compilation of blackwell_compat.cuh on Ampere */
#include "include/blackwell_compat.cuh"
#include <stdio.h>

__global__ void test_kernel() {
    /* Test NVFP4Block construction */
    NVFP4Block block;

    /* Test tensor scale */
    NVFP4TensorScale ts(2.0f);

    /* Test TMEM mock functions (should be no-ops on Ampere) */
    void* ptr = nullptr;
    tmem_alloc(&ptr, 1024);
    tmem_free(ptr);

    /* Test alignment check */
    float aligned_data[4] __attribute__((aligned(16)));
    bool is_aligned = is_aligned_128bit(aligned_data);
    (void)is_aligned;
}

int main() {
    printf("Blackwell compatibility header compiled successfully on Ampere!\n");
    printf("NVFP4Block size: %zu bytes\n", sizeof(NVFP4Block));
    printf("Two-level scaling ready for NVFP4 challenge.\n");
    return 0;
}
