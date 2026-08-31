---
name: guimlab-cuda-forge
description: "Bare-metal CUDA PTX, Tensor Cores MMA, warp-level primitives, and zero-alloc memory management guidelines for GuimLab."
---

# GuimLab CUDA Forge & Kernel Authoring Doctrine

Guidelines for authoring, refactoring, and optimizing GPU kernels within GuimLab.

## 1. Register & Memory Hierarchy Rules
- **L0 Reflex Core**: Must reside entirely in GPU registers (32x32 max size). Zero VRAM access permitted during the tick.
- **Shared Memory (`__shared__`)**:
  - Always enforce 32-bank conflict-free stride layouts.
  - Use padding (e.g. `float s_mem[32][33]`) when storing 2D warp matrices.
- **Tensor Cores (`mma.sync`)**:
  - Target native PTX instructions `mma.sync.aligned.m16n8k16.row.col` for FP16/BF16 matrix multiply-accumulate in the Lovelace Cortex.
  - Preserve FP32 accumulation registers for numerical stability in synaptic trace integration.

## 2. Warp-Level Primitives & Sparse Execution (DDWR)
- **Cooperative Warps**: Use `__ballot_sync`, `__shfl_xor_sync`, and `__activemask()` instead of block-wide barriers (`__syncthreads()`) on the critical path.
- **Delta-Driven Warp Routing (DDWR)**:
  - If a thread detects an input delta below $\epsilon$, mask out the synaptic update via `__ballot_sync`.
  - When all threads in a warp are masked out, bypass VRAM reads for traces and weights entirely.

## 3. Zero-Allocation Mandate
- Never invoke `cudaMalloc`, `cudaFree`, `cudaMallocManaged`, or `cudaMemcpy` (synchronous) during continuous ticks.
- All device buffers must be allocated once during engine initialization and mapped directly into CUDA streams.
- Use `cudaStreamSynchronize` or lightweight `cudaEvent_t` for stream fences.
