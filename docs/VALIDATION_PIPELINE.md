# GuimLab Validation Pipeline & Quality Gates

The 6-stage validation pipeline required for every candidate change before merging or deploying.

---

## 1. Stage Overview

```
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │ Stage 1:     │ ──► │ Stage 2:     │ ──► │ Stage 3:     │
 │ Static Audit │     │ C++ Unit     │     │ Parity CPU   │
 └──────────────┘     └──────────────┘     └──────────────┘
                                                  │
 ┌──────────────┐     ┌──────────────┐            │
 │ Stage 6:     │ ◄── │ Stage 5:     │ ◄── ┌──────┴───────┐
 │ Real-Time RTT│     │ Memory Leak  │     │ Stage 4:     │
 │ Latency      │     │ Sentinel (0B)│     │ Full-Frame   │
 └──────────────┘     └──────────────┘     └──────────────┘
```

1. **Stage 1: Compile-Time & Invariant Proof Gate**
   - Run native C++20 integrity and invariant suite:
     ```bash
     ctest --preset all-tests -R HarnessIntegrityTest --output-on-failure
     ```
   - Validates 64-byte cache alignment, $\det(R) = 1.0 \pm 10^{-6}$ unitary symplectic rotation, monotonic continuous-time trace decay, and TMD-ET bounds.

2. **Stage 2: C++ Unit Test Matrix**
   - Execute GoogleTest test suite:
     ```bash
     ctest --preset all-tests --output-on-failure
     ```
   - Covers `test_correctness`, `test_latency`, `test_plasticity`, `test_convergence`, `test_memory`, `test_episodic_vsa`, `test_global_workspace`, `test_world_model`, `test_symplectic_discovery`, `test_tensor_cores_cortex`, and `test_studio_harness`.

3. **Stage 3: Numerical Parity against Scalar Reference**
   - Compare CUDA kernel execution output against scalar reference (`src/guim_cpu_ref.cpp`) over $10^4$ frames.
   - Max absolute error threshold: $\le 10^{-6}$.

4. **Stage 4: Full-Frame Invariance Gate**
   - Verify unedited sensor channels and neighboring memory buffers remain byte-identical after each frame tick.

5. **Stage 5: VRAM Memory Leak Sentinel**
   - Continuous test run over $10^5$ frames with zero memory reallocation.
   - Assert Resident VRAM delta is strictly $0.00\,\text{B}$.

6. **Stage 6: Real-Time RTT Latency Assertion**
   - Verify median RTT $\le 350\,\mu\text{s}$ and kernel latency $\le 5.0\,\mu\text{s}$.
