# Guimlab — Benchmark Harness

A reproducible benchmark & correctness suite for the fused CUDA continual
learning kernel (`src/guim_kernel.cu`).  Targets:

- **Latency**: < 50 µs / frame on RTX 40xx-class GPUs
- **Throughput**: ≥ 60 fps real-time
- **Convergence**: RTRL-driven TD-error decay on a sine-prediction task
- **Memory stability**: no leaks, no NaN/Inf over 1M-frame runs
- **Plasticity**: CBP-style reset counter monotonicity and stability under
  pathological inputs

## Layout

```
tests/
├── CMakeLists.txt         # CMake build (nvcc + GoogleTest)
├── bench_common.h         # tiny self-contained helpers (percentile, JSON)
├── bench_main.cpp         # standalone microbench orchestrator
├── test_correctness.cpp   # gtest: construct/destroy + NaN/Inf contract
├── test_latency.cpp       # gtest: soft per-frame latency bound
├── test_convergence.cpp   # gtest: sine-prediction learning trace
├── test_plasticity.cpp    # gtest: reset-counter monotonicity
└── test_memory.cpp        # gtest: 1M-frame leak + finiteness
```

The C ABI in `include/guim_kernel.h` is the source of truth for constants;
the modern C++ wrapper `include/guim/guim.h` provides an RAII `Engine` class
used by the tests.

## Build

```bash
# from the project root
cmake -S tests -B build/tests -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/tests -j
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `89` (Ada Lovelace / RTX 40xx).  For
other GPUs:

| Arch | GPU |
|------|-----|
| `70` | Volta   (V100, Titan V) |
| `75` | Turing  (RTX 20xx, T4) |
| `80` | Ampere  (RTX 30xx, A100) |
| `86` | Ampere  (RTX 30xx laptop) |
| `89` | Ada     (RTX 40xx, L4) |
| `90` | Hopper  (H100) |

## Run

### All tests (gtest)

```bash
./build/tests/guim_kernel_tests
./build/tests/guim_kernel_tests --gtest_filter='GuimLatency.*'
```

### Microbench orchestrator

```bash
./build/tests/guim_bench --frames 10000 --warmup 500
./build/tests/guim_bench --json build/bench_results.json
```

The JSON file contains the structure documented in the project README:

```json
{
  "latency_us": {"avg": ..., "p50": ..., "p99": ..., "max": ...},
  "throughput_fps": ...,
  "convergence_td_error_final": null,
  "memory_used_mb": ...,
  "plasticity_resets_count": ...,
  "nan_inf_detected": false
}
```

`convergence_td_error_final` is `null` in staging mode (placeholder kernel
emits NaN) and will be populated once the fused kernel lands.

## Staging caveats (2026-08-31)

The current `guim_kernel.cu` is a **placeholder** that:

1. Allocates one GPU arena up-front (size documented in
   `Engine::bytes_resident()`).
2. Returns NaN on every step to signal "kernel not yet implemented".
3. Increments `resets_count()` on NaN/Inf *input* (input guard).

Because of (2), the convergence and memory-stability tests use
`GTEST_SKIP` when they detect the NaN sentinel.  The remaining tests
(construct/destroy, monotonicity, arena stability, cudaMemGetInfo
leak check) pass unchanged and exercise the full build pipeline end-to-end.

Once `fused_continual_step_kernel` (or equivalent) lands, every assertion
becomes live without further changes to the harness.

## Adapting the harness when the kernel lands

1. Remove the `GTEST_SKIP` clauses in `test_memory.cpp` and
   `test_convergence.cpp`.
2. In `test_convergence.cpp::GuimConvergence.HiddenStaysFiniteOver10kFrames`,
   add an assertion that the TD-error magnitude at frame `kFrames` is less
   than 50% of the magnitude at frame 0.
3. In `bench_main.cpp`, capture the final TD-error magnitude and emit it
   instead of `null` in `convergence_td_error_final`.
4. In `test_latency.cpp`, lower the soft bound from `5'000 us` to `50 us`
   once the real kernel is in place.

## Hyperparameter reference

All constants live in `include/guim_kernel.h`:

```c
#define GUIM_STATE_DIM           64
#define GUIM_INPUT_DIM           32
#define GUIM_TOTAL_DIM           (GUIM_STATE_DIM + GUIM_INPUT_DIM)  // 96
#define GUIM_TOTAL_W             (GUIM_STATE_DIM * GUIM_TOTAL_DIM)  // 6144
#define GUIM_PLASTICITY_WINDOW   256
#define GUIM_ALPHA               0.005f
#define GUIM_LAMBDA              0.85f
#define GUIM_WEIGHT_DECAY        0.0001f
#define GUIM_PLASTICITY_THRESHOLD 0.001f
#define GUIM_WEIGHT_CLIP         5.0f
#define GUIM_TRACE_CLIP          100.0f
```

Rationale and sources are documented in
`docs/01-foundations.md` (planned) and the project `README.md`.

## References

- Khurram Javed et al., *Continual Backpropagation* (2024) — plasticity
  preservation via per-neuron variance threshold + re-randomisation.
- Richard Sutton, *The Bitter Lesson* / *TD(λ)* (1988) — eligibility
  traces, credit assignment over time.
- John Carmack, *Arena Allocation* — single allocation at boot, zero
  dynamic allocation in the hot path.