# NVFP4 GEMM Memory Map: Allocation-Free Architecture

## Overview

This document defines the memory hierarchy and data flow for our Blackwell-optimized
NVFP4 GEMM kernel. The key innovation is **zero runtime allocations** - all memory
is statically mapped at compile time, with data flowing through TMA → SMEM → TMEM → Tensor Cores.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY HIERARCHY OVERVIEW                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Global Memory (HBM3e)                                                      │
│   ├── Matrix A: [M, K] in NVFP4 format (packed with E4M3 scales)            │
│   ├── Matrix B: [K, N] in NVFP4 format (packed with E4M3 scales)            │
│   ├── Matrix C: [M, N] output in FP16/BF16                                  │
│   └── Tensor Scales: [num_tiles] FP32 (Level-2 scales)                      │
│           │                                                                  │
│           │ TMA Descriptor (async, non-blocking)                            │
│           ▼                                                                  │
│   Shared Memory (228 KB per SM on Blackwell)                                │
│   ├── SMEM_A: Staging buffer for A tiles                                    │
│   ├── SMEM_B: Staging buffer for B tiles                                    │
│   └── SMEM_SCALES: E4M3 micro-block scales                                  │
│           │                                                                  │
│           │ tcgen05.ld (TMEM load)                                          │
│           ▼                                                                  │
│   Tensor Memory - TMEM (256 KB per SM, Blackwell-only)                      │
│   ├── TMEM_A: Operand A in tensor-ready format                              │
│   ├── TMEM_B: Operand B in tensor-ready format                              │
│   └── TMEM_ACCUM: FP32 accumulator (replaces register accumulation!)        │
│           │                                                                  │
│           │ tcgen05.mma (Tensor Core execution)                             │
│           ▼                                                                  │
│   Tensor Cores (5th Gen on Blackwell)                                       │
│   └── Native NVFP4 × NVFP4 → FP32 accumulation                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Tiling Strategy: 128×128 Output Tile

### 1.1 Tile Dimensions

We target a **128×128 output tile** per thread block, computed via a sequence of
**K-dimension reductions** using 64-element K-slices.

```
Output Tile:  C[128, 128]  (one per thread block)

A Tile:       A[128, 64]   (M=128, K_slice=64)
B Tile:       B[64, 128]   (K_slice=64, N=128)

K-loop iterations: K_total / 64
```

### 1.2 NVFP4 Packing Layout

Each NVFP4Block contains 16 FP4 elements + 1 E4M3 scale (16 bytes aligned).

```
A Tile [128, 64] in NVFP4:
├── 128 rows × 64 columns = 8,192 FP4 elements
├── 8,192 / 16 = 512 NVFP4Blocks
├── 512 × 16 bytes = 8,192 bytes (8 KB) for data + scales
└── Layout: Row-major, 4 blocks per row (64 elements / 16 = 4)

B Tile [64, 128] in NVFP4:
├── 64 rows × 128 columns = 8,192 FP4 elements
├── 8,192 / 16 = 512 NVFP4Blocks
├── 512 × 16 bytes = 8,192 bytes (8 KB)
└── Layout: Column-major for coalesced access during MMA
```

### 1.3 Thread Block Configuration

```
Block dimensions:  128 threads (4 warps)
Warp arrangement:  2×2 warp tile (each warp owns 64×64 of output)

Warp 0: C[0:64,   0:64]
Warp 1: C[0:64,   64:128]
Warp 2: C[64:128, 0:64]
Warp 3: C[64:128, 64:128]
```

---

## 2. TMA Loading: Global → Shared Memory

### 2.1 TMA Descriptor Setup (Host-Side)

TMA (Tensor Memory Accelerator) enables **asynchronous, address-generation-free**
loads from global memory. We define descriptors at kernel launch:

```cpp
// Host-side TMA descriptor creation
CUtensorMap tma_desc_A, tma_desc_B;

// A matrix: [M, K] with 128×64 tiles
cuTensorMapEncodeTiled(
    &tma_desc_A,
    CU_TENSOR_MAP_DATA_TYPE_UINT8,  // FP4 packed as bytes
    2,                               // 2D tensor
    global_A,                        // Base pointer
    {K/2, M},                        // Dimensions (K/2 because 2 FP4 per byte)
    {K/2, 1},                        // Strides
    {32, 128},                       // Box dimensions (64 FP4 = 32 bytes, 128 rows)
    {1, 1},                          // Element strides
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,      // 128-byte swizzle for bank conflict avoidance
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

### 2.2 Async TMA Load (Device-Side)

```cpp
__device__ void load_tile_via_tma(
    void* smem_ptr,
    const CUtensorMap* tma_desc,
    int tile_row,
    int tile_col
) {
    // Only one thread per warp issues TMA
    if (threadIdx.x % 32 == 0) {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :
            : "r"(smem_ptr), "l"(tma_desc), "r"(tile_col), "r"(tile_row), "r"(mbar_ptr)
            : "memory"
        );
    }
}
```

### 2.3 Shared Memory Layout

```
Shared Memory Map (total: ~49 KB per tile pair + double buffer)
═══════════════════════════════════════════════════════════════

Offset 0x0000 - 0x1FFF:   SMEM_A_0 [8 KB]   ─┐
Offset 0x2000 - 0x3FFF:   SMEM_B_0 [8 KB]    ├─ Buffer 0 (current)
Offset 0x4000 - 0x41FF:   SMEM_SCALES_0 [512 bytes] ─┘

Offset 0x4200 - 0x61FF:   SMEM_A_1 [8 KB]   ─┐
Offset 0x6200 - 0x81FF:   SMEM_B_1 [8 KB]    ├─ Buffer 1 (prefetch)
Offset 0x8200 - 0x83FF:   SMEM_SCALES_1 [512 bytes] ─┘

Offset 0x8400 - 0x87FF:   Barriers & metadata [1 KB]

Total: ~34 KB (well within 228 KB limit)
```

---

## 3. SMEM → TMEM Transition via tcgen05

### 3.1 TMEM Organization (Blackwell)

TMEM is organized as a **2D register file extension**:
- 256 KB per SM
- Arranged in rows × columns (logical 2D addressing)
- Direct feed to Tensor Cores without register file pressure

```
TMEM Layout for 128×128 tile:
═════════════════════════════

TMEM_A:     [128, 64] → 8,192 FP4 elements (unpacked to FP16 for MMA)
TMEM_B:     [64, 128] → 8,192 FP4 elements
TMEM_ACCUM: [128, 128] → 16,384 FP32 elements = 64 KB

Total TMEM usage: ~72 KB per warp-group (fits in 256 KB)
```

### 3.2 TMEM Load from Shared Memory

```cpp
__device__ void smem_to_tmem_load(
    uint32_t tmem_addr,     // TMEM destination (opaque handle)
    const void* smem_ptr,   // Shared memory source
    uint32_t size_bytes
) {
    // tcgen05.ld - Load from SMEM to TMEM
    asm volatile(
        "tcgen05.ld.sync.aligned.16x64b.x4.b32 [%0], [%1];"
        :
        : "r"(tmem_addr), "l"(smem_ptr)
        : "memory"
    );
}
```

### 3.3 TMEM-Based MMA Execution

The critical advantage: **accumulation stays in TMEM**, not registers!

```cpp
__device__ void tmem_mma_fp4(
    uint32_t tmem_accum,    // TMEM accumulator address
    uint32_t tmem_a,        // TMEM operand A
    uint32_t tmem_b,        // TMEM operand B
    float tensor_scale_a,   // Level-2 scale for A
    float tensor_scale_b    // Level-2 scale for B
) {
    // tcgen05.mma with NVFP4 operands, FP32 accumulation in TMEM
    // Scale factors applied during accumulation
    asm volatile(
        "tcgen05.mma.sync.aligned.m64n64k64.f32.e2m1.e2m1.f32 "
        "[%0], [%1], [%2], [%0], %3, %4;"
        :
        : "r"(tmem_accum), "r"(tmem_a), "r"(tmem_b),
          "f"(tensor_scale_a * tensor_scale_b), "r"(0)
        : "memory"
    );
}
```

---

## 4. Complete Data Flow (One K-Iteration)

```
Step 1: TMA Prefetch (async, overlapped with compute)
────────────────────────────────────────────────────
  Global[A_tile_next] ──TMA──► SMEM_A_1 (buffer 1)
  Global[B_tile_next] ──TMA──► SMEM_B_1 (buffer 1)

Step 2: SMEM → TMEM Load (current tile from buffer 0)
────────────────────────────────────────────────────
  SMEM_A_0 ──tcgen05.ld──► TMEM_A
  SMEM_B_0 ──tcgen05.ld──► TMEM_B

Step 3: Scale Extraction
────────────────────────────────────────────────────
  SMEM_SCALES_0 ──► Registers (E4M3 scales, small footprint)

Step 4: Tensor Core MMA (accumulate in TMEM)
────────────────────────────────────────────────────
  TMEM_A × TMEM_B ──tcgen05.mma──► TMEM_ACCUM += result
  (Two-level scaling applied: E4M3 micro × FP32 tensor)

Step 5: Barrier & Buffer Swap
────────────────────────────────────────────────────
  Wait for TMA completion on buffer 1
  Swap buffer pointers: 0 ↔ 1

Step 6: Repeat for next K-slice
```

---

## 5. Shared Memory Bank Conflict Analysis

### 5.1 Bank Layout (Blackwell)

Shared memory has **32 banks**, each 4 bytes wide. Consecutive 4-byte words
map to consecutive banks.

```
Address:  0x00  0x04  0x08  0x0C  ...  0x7C  0x80  0x84
Bank:       0     1     2     3   ...   31     0     1
```

### 5.2 NVFP4Block Access Pattern

Each NVFP4Block is 16 bytes (4 words). When 32 threads in a warp access
consecutive blocks:

```
Thread 0:  Block 0  → addresses 0x00-0x0F → banks 0,1,2,3
Thread 1:  Block 1  → addresses 0x10-0x1F → banks 4,5,6,7
Thread 2:  Block 2  → addresses 0x20-0x2F → banks 8,9,10,11
...
Thread 7:  Block 7  → addresses 0x70-0x7F → banks 28,29,30,31
Thread 8:  Block 8  → addresses 0x80-0x8F → banks 0,1,2,3  ← CONFLICT!
```

### 5.3 POTENTIAL BANK CONFLICT IDENTIFIED

**Problem:** With 16-byte blocks and 32 banks, threads 0 and 8 (and 16, 24)
access the same banks simultaneously.

**Pattern:** Every 8 threads collide → **4-way bank conflict** per access.

### 5.4 Proposed Solutions

#### Option A: 128-byte Swizzle (Recommended)

Use TMA's built-in swizzle mode to permute addresses:

```cpp
CU_TENSOR_MAP_SWIZZLE_128B  // In TMA descriptor
```

This XORs address bits to distribute accesses across banks:
```
Swizzled address = original_addr ^ ((original_addr >> 7) & 0x7F)
```

#### Option B: Padding

Add 16 bytes of padding per 128-byte row:

```cpp
// Instead of 8 KB for 512 blocks:
// Use 8 KB + (512/8)*16 = 9 KB with padding
__shared__ char smem_A[9216];  // 8 KB + 1 KB padding
```

#### Option C: Software XOR Addressing

Compute permuted indices in the kernel:

```cpp
int swizzled_idx = block_idx ^ ((block_idx >> 3) & 0x7);
NVFP4Block* ptr = &smem_base[swizzled_idx];
```

---

## 6. Register Budget Analysis

With TMEM handling accumulation, register pressure drops dramatically:

```
Traditional GEMM (register accumulation):
─────────────────────────────────────────
  Accumulator: 128×128 / 32 threads = 512 FP32 values = 512 registers ❌
  (Exceeds 255 register limit!)

TMEM-Based GEMM (our approach):
─────────────────────────────────────────
  Accumulator: In TMEM, not registers = 0 registers ✓
  Operand staging: ~16 registers
  Loop indices: ~8 registers
  Scale factors: ~8 registers
  Pointers: ~8 registers
  ─────────────────────────────
  Total: ~40 registers ✓ (well under 128 limit)
```

---

## 7. Audit Request

**Please review the following for potential issues:**

1. **Bank Conflict Mitigation:** Is the 128-byte swizzle sufficient, or should
   we implement software XOR addressing as a fallback for non-TMA paths?

2. **TMEM Allocation Size:** At 72 KB per tile, can we safely run 3 warp-groups
   per SM (216 KB < 256 KB), or should we reduce tile size for higher occupancy?

3. **Double Buffering Depth:** Is 2-deep buffering sufficient to hide TMA latency,
   or should we implement 3-deep (triple) buffering at the cost of more SMEM?

4. **Scale Factor Handling:** The E4M3 micro-scales are currently loaded to
   registers. Should we instead keep them in SMEM and load on-demand to reduce
   register pressure further?

---

## 8. Summary: Why This Wins

| Traditional GEMM | Our Allocation-Free TMEM GEMM |
|------------------|-------------------------------|
| cudaMalloc for workspace | Zero allocations |
| Register accumulation (512 regs) | TMEM accumulation (0 regs) |
| ~25% occupancy | ~75%+ occupancy target |
| Manual prefetch scheduling | TMA hardware prefetch |
| Bank conflicts from naive layout | 128B swizzle elimination |
| Two memory hops (Global→Reg) | Three hops but async (Global→SMEM→TMEM) |

The TMEM accumulator is the key innovation. By moving the 64 KB accumulator
out of the register file and into TMEM, we unlock dramatically higher
occupancy while maintaining the numerical precision of FP32 accumulation.
