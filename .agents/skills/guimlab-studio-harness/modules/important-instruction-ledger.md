# Important Instruction Ledger & Constitutional Lock

Immutable rules that must be preserved without exception across all agent sessions.

## Constitutional Invariants
1. **100% Native C++20 & Bare-Metal CUDA**: Zero Python in inference, zero PyTorch, LibTorch, cuDNN, or ONNX Runtime.
2. **Zero Dynamic Allocation on Critical Path**: Static memory arenas and lock-free cache-aligned ring buffers only.
3. **No CUDA Graphs on Persistent Kernels**: Persistent kernels with spin-wait loops (`while (!terminate)`) must never be wrapped in CUDA Graphs.
4. **No Non-Symplectic Destabilization**: The symplectic phase space $(E_{ij}, P_{ij})$ and Kuramoto frequency locks must never be altered with unconstrained ODEs.
5. **No Warp-Divergent KAN on Reflex L0**: The Reflex L0 core must remain single-warp (`__launch_bounds__(32, 1)`), zero-divergent, with 0 byte register spill.
6. **No Blocking Slow-Wave LLM Coupling**: External models or discrete LLMs must be strictly decoupled and asynchronous.
