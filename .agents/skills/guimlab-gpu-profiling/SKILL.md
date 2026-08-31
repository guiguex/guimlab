---
name: guimlab-gpu-profiling
description: "Nanosecond/microsecond GPU profiling, NVTX v3 instrumentation, Nsight Compute / Systems benchmarks for GuimLab."
---

# GuimLab GPU Profiling & Performance Workstation

Protocol for cycle-accurate profiling, latency attribution, and hardware roofline validation.

## 1. Instrumentation Standards
- **NVTX v3 Ranges**: All high-level steps (Reflex step, Lovelace Cortex step, Plasticity Sweeper, SHM I/O) must be demarcated with `nvtx3::scoped_range`.
- **Hardware Timers**: Use `__rdtsc()` on CPU and `cudaEventElapsedTime()` on GPU.
- **Latency Attribution Budget**:
  - L0 Reflex Tick: Budget < 200 ns.
  - L1/L2 Lovelace Cortex Tick: Budget < 5.0 µs.
  - Host-to-Device IPC Round-Trip: Budget < 400 µs (p99).

## 2. Profiling Command Matrix
```bash
# Detailed Kernel Analysis (SM Occupancy, Register Pressure, Memory Roofline)
ncu --set full --target-processes all ./build/bin/guim_bench

# Systems-level Timeline Trace (Streams, NVTX, IPC Synchronization)
nsys profile --trace=cuda,nvtx,osrt --sample=cpu --output=guim_trace ./build/bin/stream_wav_demo
```

## 3. Register Spilling & Occupancy Check
- Check compiler output (`-Xptxas -v`):
  - `0 bytes spill stores, 0 bytes spill loads`.
  - Enforce max register usage with `__launch_bounds__(threads_per_block, min_blocks_per_sm)`.
