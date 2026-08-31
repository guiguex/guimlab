# GuimLab Codebase Architecture Index

Hierarchical map of all subsystems in the GuimLab AGI runtime.

---

## 1. Subsystems

### Subsystem 1: Reflex L0 Core (KISS Engine)
* **Directory**: `src/kiss_reflex_kernel.cu`, `include/guim_kernel.h`
* **Invariants**: 32x32 single-warp (`__launch_bounds__(32, 1)`), 0 byte spill, < 100 ns execution.

### Subsystem 2: Lovelace Cortex L1/L2
* **Directory**: `src/guim_lovelace_kernel.cu`, `include/lovelace_cortex.h`
* **Invariants**: Tensor Cores `mma.sync`, Continuous-Time Traces (CF-TT), TMD-ET adaptive meta-learning, < 5.0 µs execution.

### Subsystem 3: Symplectic Riemannian Momentum Traces (SR-MIT)
* **Directory**: `src/symplectic_traces_kernel.cu`, `include/symplectic_traces.h`
* **Invariants**: 2-form phase space $(E_{ij}, P_{ij}) \in \mathbb{R}^2$, Kuramoto harmonic phase-lead compensation.

### Subsystem 4: Continual Backpropagation (CBP) & Plasticity Sweeper
* **Directory**: `src/plasticity_sweeper.cu`, `include/guim_kernel.h`
* **Invariants**: Running metabolic variance $\sigma_i^2$, in-place asynchronous neurogenesis.

### Subsystem 5: Zero-Copy Lock-Free IPC & DDWR
* **Directory**: `include/ipc_structs_v2.h`, `include/ipc_structs_sparse.h`, `src/delta_filter.cpp`
* **Invariants**: Cache-aligned (`alignas(64)`), AVX2 delta scanning, $>90\%$ VRAM bandwidth reduction.

### Subsystem 6: GuimLab Studio (Visual Cockpit)
* **Directory**: `src/studio/`, `include/guim_studio.h`, `include/guim_viewport_session.h`
* **Invariants**: Pure C++20 Dear ImGui / ImPlot, 1000 FPS, zero dynamic allocations on frame loop.

### Subsystem 7: Agent Harness & Optimization Engine
* **Directory**: `.agents/skills/guimlab-studio-harness/`, `scripts/`
* **Invariants**: 7-stage validation pipeline, Proof Contract enforcement, NVTX profiling.
