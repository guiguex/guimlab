# CUDA Kernel Authoring & Bare-Metal Engineering

Guidelines for high-performance neuromorphic GPU kernel implementation in GuimLab.

## 1. Register Allocation & Launch Bounds
- **Reflex L0 Kernel**: Enforce `__launch_bounds__(32, 1)`. The kernel operates within a single warp with 0 spill stores/loads.
- **Lovelace Cortex Kernel**: Enforce `__launch_bounds__(128, 2)` for Tensor Core tile operations (`mma.sync.aligned.m16n8k16`).
- Verify PTX compilation statistics using `-Xptxas -v` and fail any build introducing spills into local memory.

## 2. Warp-Level Primitives & Memory Layout
- Use `__ballot_sync` and `__shfl_xor_sync` for warp-wide reduction without shared memory overhead.
- Delta-Driven Warp Routing (DDWR):
  ```cpp
  unsigned int active_mask = __ballot_sync(__activemask(), fabsf(delta_input[lane_id]) > EPSILON);
  if (active_mask == 0) {
      // Warp is dormant, skip VRAM trace and weight load completely
      return;
  }
  ```
- Structure shared memory (`__shared__`) to eliminate 32-bank conflicts by padding 2D matrices (e.g. `float s_tile[32][33]`).

## 3. Numerical Stability & Floating-Point Discipline
- Preserve FP32 accumulation in Tensor Cores when performing synaptic updates.
- Clamp meta-learning rates $\beta_{ij} \in [\beta_{min}, \beta_{max}]$ and eligibility traces to prevent gradient explosion.
- Use native CUDA intrinsics: `__expf()`, `__fdividef()`, `__saturatef()`.
