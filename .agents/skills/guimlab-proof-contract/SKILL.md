---
name: guimlab-proof-contract
description: "Non-regression verification protocol, mathematical parity invariants, and automated proof contracts for GuimLab."
---

# GuimLab Proof Contract & Non-Regression Gatekeeper

Every change to the core kernels, IPC memory layouts, or mathematical dynamics must pass the **Proof Contract Gate**.

## 1. Required Proof Metrics Table
Before closing any task or proposing an optimization, generate the standard verification table:

| Metric | Baseline | Candidate | Delta | Status |
| :--- | :--- | :--- | :--- | :--- |
| **GPU Kernel Latency (µs)** | e.g. 3.8 µs | e.g. 3.6 µs | -0.2 µs | ✅ PASS |
| **Host-Device RTT p99 (µs)** | e.g. 308.5 µs | e.g. 305.0 µs | -3.5 µs | ✅ PASS |
| **Max Absolute Parity Error** | $\le 10^{-6}$ | $\le 10^{-6}$ | 0.0 | ✅ PASS |
| **Spill Stores / Loads** | 0 B / 0 B | 0 B / 0 B | 0 B | ✅ PASS |
| **VRAM Leaks (10^5 frames)** | 0.00 B | 0.00 B | 0 B | ✅ PASS |
| **Dead Unit Recovery Rate** | 100% | 100% | 0% | ✅ PASS |

## 2. Invariant Rules & Anti-Pattern Gatekeepers
1. **Zero Fake Signals**: Tests must not pass on silent zero tensors, default uninitialized structs, or missing error codes.
2. **Deterministic Seed Parity**: Scalar reference (`guim_cpu_ref.cpp`) and CUDA implementation must produce identical output states over $10^4$ frames given identical input streams.
3. **Full-Frame Invariance**: External state (memory outside edited buffers) must remain byte-identical before and after the step.
4. **No CUDA Graphs on Persistent Kernels**: Persistent kernels with IPC spin-wait loops (`while (!terminate)`) must never be wrapped in CUDA Graphs.
5. **No Non-Symplectic Destabilization**: The 2-form phase space $(E_{ij}, P_{ij})$ and Kuramoto frequency locks must never be replaced by non-Hamiltonian or unconstrained ODEs.
6. **No Warp-Divergent KAN on Reflex L0**: The L0 Reflex Core must remain single-warp (`__launch_bounds__(32, 1)`), zero-divergent, with 0 byte register spill.
7. **No Blocking Slow-Wave LLM Coupling**: Any external tool or discrete LLM bridge must be strictly decoupled, non-blocking, and asynchronous.

