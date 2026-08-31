# GPU Profiling Workstation & Roofline Analysis

Guidelines for cycle-accurate profiling, NVTX instrumentation, and latency attribution.

## 1. NVTX v3 Protocol
Encapsulate all core execution blocks with zero-overhead NVTX v3 scoped ranges:
```cpp
#include <nvtx3/nvtx3.hpp>

void LovelaceCortexEngine::step(...) {
    nvtx3::scoped_range r{"Lovelace_Cortex_Step"};
    // Kernel launch & execution
}
```

## 2. Profiling & Benchmarking Commands
```bash
# Full Kernel Profiling (SM Occupancy, Instructions, Roofline)
ncu --set full --target-processes all ./build/bin/guim_bench

# System Timeline (Host-Device RTT, Stream Synchronization, NVTX)
nsys profile --trace=cuda,nvtx,osrt --sample=cpu --output=guim_timeline ./build/bin/stream_wav_demo
```

## 3. Strict Acceptance Criteria
- isolated GPU kernel latency $\le 5.0\,\mu\text{s}$ (L1/L2 Cortex) and $\le 200\,\text{ns}$ (Reflex L0).
- Median Host-to-Device RTT $\le 350\,\mu\text{s}$ ($p_{50}$), $p_{99} \le 450\,\mu\text{s}$.
- Zero VRAM memory leaks over $10^5$ continuous frames.
- Minimum 80% theoretical SM active occupancy on Ampere/Ada GPUs (`sm_86`/`sm_89`).
