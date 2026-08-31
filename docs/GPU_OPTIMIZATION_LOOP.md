# GuimLab GPU Optimization Loop & Roofline Protocol

Protocol for optimizing CUDA kernels, Tensor Cores matrix multiplications, and memory layouts in GuimLab.

## 1. The 7-Step Optimization Loop

```
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  1. Baseline │ ──► │  2. Roofline │ ──► │  3. Isolate  │ ──► │ 4. Transform │
  │    Metrics   │     │   Diagnosis  │     │  Bottleneck  │     │   (PTX/SM)   │
  └──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                        │
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
  │  7. Proof    │ ◄── │  6. Benchmark│ ◄── │ 5. Numerical │ ◄──────────┘
  │   Contract   │     │   Speedup    │     │    Parity    │
  └──────────────┘     └──────────────┘     └──────────────┘
```

1. **Baseline Metrics**: Record execution time ($\mu\text{s}$), register count (`ptxas`), SM occupancy %, and VRAM bandwidth.
2. **Roofline / SOL Diagnosis**: Determine whether the kernel is memory-bound (VRAM/L2) or compute-bound (Tensor Cores / ALU).
3. **Isolate Top Bottleneck**: Profile one specific kernel or warp bottleneck at a time (e.g. bank conflict, warp divergence, register spilling).
4. **Transform Kernel**: Apply targeted PTX intrinsics, Tensor Core tiling (`mma.sync`), or DDWR warp masking.
5. **Numerical Parity**: Validate against the scalar CPU reference (`guim_cpu_ref.cpp`) over $10^4$ frames ($\text{error} \le 10^{-6}$).
6. **Benchmark Speedup**: Confirm isolated kernel execution latency reduction and host-to-device round-trip latency.
7. **Proof Contract**: Commit only with an updated `Proof Contract` table documenting the delta.

---

## 2. Native Benchmarking & Profiling Protocol

Execute the optimization loop using pure native CMake presets and Nsight tools:

```bash
# 1. Compile with Benchmark / Release preset
cmake --preset release
cmake --build --preset release --target guim_bench guim_tests -j

# 2. Run Numerical Parity & Invariant Proofs
ctest --preset all-tests --output-on-failure

# 3. Microsecond Kernel Profiling with Nsight Compute
ncu --set full --target-processes all ./build/release/bin/guim_bench

# 4. Systems-Level Host-Device Timeline Trace with Nsight Systems
nsys profile --trace=cuda,nvtx,osrt --sample=cpu --output=guim_timeline ./build/release/bin/stream_wav_demo
```
