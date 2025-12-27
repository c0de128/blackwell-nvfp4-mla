# CLAUDE.md - NVIDIA Blackwell Performance Challenge (NVFP4)

## Project Strategy
- **Primary Goal:** Winning entry for optimizing GEMM/Attention kernels via NVFP4 on GB200.
- **Architectural Edge:** Port "allocation-free" MLA (Multi-Head Latent Attention) logic.
- **Key Innovation:** Leverage Blackwell-specific TMEM (Tensor Memory) to bypass register pressure and use 1x16 micro-block scaling for FP4 precision.

## Hardware & Environment
- **Local Dev:** NVIDIA RTX 3060 OC (Ampere/sm_86). Used for logic, syntax, and compilation.
- **Target Target:** NVIDIA GB200 (Blackwell/sm_100 or sm_90a).
- **Tooling:** Claude Code CLI for all file operations, git management, and terminal execution.

## Build & Validation Commands
- **Compiling for Blackwell (PTX):** `nvcc -ptx -arch=sm_90a -O3 kernel.cu`
- **Register Usage Audit:** `nvcc -Xptxas -v -arch=sm_86 kernel.cu`
- **Logic Validation:** `nvcc -arch=sm_86 -O3 kernel.cu -o test_bench && ./test_bench`
- **Linting:** `cppcheck --enable=all --suppress=missingIncludeSystem .`

## Mandatory Coding Standards
- **Allocation-Free:** Zero `cudaMalloc` calls within the hot path. Use static shared memory or TMEM.
- **Precision:** Use `nvfp4` (E2M1) format. Every 16-element block must have a corresponding E4M3 scale factor.
- **Asynchronous Ops:** Mandatory use of `cp.async` for global-to-shared data movement.
- **Blackwell Primitives:** Use PTX inline assembly for `tcgen05` (TMEM) and `mma.sync` instructions that are not yet in the high-level CUDA headers.
- **Memory Alignment:** All global and shared memory accesses must be 128-bit aligned to maximize bandwidth.

## Git & PR Workflow
- **Naming:** `feat/kernel-type-optimization` (e.g., `feat/attention-tmem-scaling`).
- **Commits:** Conventional Commits (e.g., `perf(attention): implement micro-block scaling for NVFP4`).
- **PR Audit Requirements:** 1. Confirm register count <= 128 per thread.
    2. Verify no shared memory bank conflicts via code analysis.
    3. Ensure 100% test coverage on logic verification scripts.

## Senior Advisor Guardrails
- Claude must provide a technical plan (using "think hard") before modifying existing kernels.
- Claude must report register usage after every successful compilation.
- If register pressure is high, Claude must prioritize refactoring to TMEM.