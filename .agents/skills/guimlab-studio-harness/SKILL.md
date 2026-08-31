---
name: guimlab-studio-harness
description: "Master governance & engineering harness for GuimLab C++20 bare-metal neuromorphic substrate and CUDA AGI runtime."
---

# GuimLab Studio & AGI Development Harness

This skill governs all development, optimization, validation, and visual monitoring for **GuimLab** — a 100% native C++20 / CUDA neuromorphic substrate for continuous voice-to-voice cognitive agents.

## Core Directives & Constitutional Invariants

1. **100% Native C++20 & Pure CUDA**:
   - Zero runtime dependencies (No PyTorch, LibTorch, cuDNN, ONNX Runtime, or Python in the inference path).
   - Direct hardware manipulation: GPU registers, PTX assembly, CPU intrinsics (AVX2/AVX-512), lock-free POSIX/Win32 Shared Memory (SHM).

2. **Zero Dynamic Allocation on Critical Path**:
   - No `malloc`, `new`, `std::vector::resize`, or `cudaMalloc` in any tick, inference, learning, or telemetry loop.
   - All memory must be pre-allocated in static arenas and 64-byte cache-aligned ring buffers (`alignas(64)`).

3. **Continuous Neuromorphic Algorithmic Rigor**:
   - **CF-TT**: Closed-form continuous-time eligibility traces $e^{-\Delta t/\tau}$.
   - **SR-MIT**: Symplectic Riemannian Momentum-Informed Traces in 2-form phase space $(E_{ij}, P_{ij})$.
   - **TMD-ET / IDBD**: Per-synapse adaptive meta-learning rate $\alpha_{ij} = \exp(\beta_{ij})$.
   - **CBP**: Continual Backpropagation with metabolic variance tracking $\sigma_i^2$ and asynchronous in-place neurogenesis.
   - **Dual-Speed Hierarchy**: Reflex L0 (< 100 ns, register-pinned) and Lovelace Cortex L1/L2 (< 5 µs, Tensor Cores `mma.sync`).

## OODA Operational Loop

- **Observe**: Inspect `guim_studio` telemetry, NVTX ranges, sub-microsecond latency ($p_{50}, p_{99}$), and Lovelace 256-neuron thermal maps.
- **Orient**: Verify mathematical invariants, detect dead units ($\sigma_i^2 < \epsilon_{plasticity}$), audit warp divergence (`__ballot_sync`).
- **Decide**: Formulate optimization or algorithmic refinements using the `guimlab-cuda-forge` and `guimlab-proof-contract` skills.
- **Act**: Implement changes, compile natively, verify zero-leak zero-regression with `ctest`, and generate certified benchmark proof tables.
