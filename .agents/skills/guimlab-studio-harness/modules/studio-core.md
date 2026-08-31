# Studio Core & Progressive Enforcement Engine

GuimLab Studio uses progressive enforcement for neuromorphic C++20 and CUDA engineering.

## 1. Active Process Model
For every work item, the agent operates in one of three states:
- **Standard**: Default for bounded, understood C++20 / CUDA kernel edits and testing.
- **Investigative**: Triggered when mathematical derivations, memory bounds, latency spikes, or numerical parity fail.
- **Governed**: Required for cross-subsystem architecture changes (e.g. altering IPC layouts `ipc_structs_v2.h`, modifying the dual-speed Reflex/Cortex boundary).
- **Recovery**: Incident state triggered when speculative patches fail or contradictory benchmarks emerge. Reverts to baseline before reapplying proof.

## 2. Base Invariants
These rules are inviolable across all states:
- 100% Native C++20 and pure CUDA. Zero Python in inference, zero external heavy runtimes.
- Zero dynamic allocation (`malloc`, `new`, `std::vector::resize`, `cudaMalloc`) in any tick, inference, or telemetry loop.
- Invariance of the Hamiltonian phase space $(E_{ij}, P_{ij})$ in SR-MIT.
- Reflex L0 is pinned to GPU registers in a single warp (`__launch_bounds__(32, 1)`), zero-divergence.
- All IPC ring buffers are 64-byte aligned (`alignas(64)`).
- Every optimization must provide a formal Proof Contract table before acceptance.
