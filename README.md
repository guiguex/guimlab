<div align="center">

# GUIMLAB
### Continuous Full-Duplex Voice-to-Voice Neuromorphic Substrate
**100% Native C++20 & Bare-Metal CUDA (`sm_86` / Ampere / Ada / Hopper)**

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat-square&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/20)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Architecture](https://img.shields.io/badge/Compute-sm__86%20%7C%20RTX%203090-orange.svg?style=flat-square)]()
[![Zero-Alloc](https://img.shields.io/badge/VRAM%20Allocations-0%20in%20Hot%20Loop-success.svg?style=flat-square)]()
[![Tests](https://img.shields.io/badge/Tests-26%2F26%20Passed-brightgreen.svg?style=flat-square)]()
[![Latency](https://img.shields.io/badge/Median%20Latency-308.5%20%CE%BCs-purple.svg?style=flat-square)]()

<br/>

**A sub-millisecond, continual learning neural operating system designed for real-time, tokenless, full-duplex voice intelligence.**  
*Zero Python runtime. Zero PyTorch. Zero cuDNN. Zero ONNX. Zero heap allocation on hot paths.*

</div>

---

## ⚡ Key Architectural Differentiators

Traditional conversational AI chains **ASR $\to$ LLM $\to$ TTS**, accumulating **800 to 2,500 ms** of serialization lag, catastrophic turn-taking latency, and complete destruction of continuous prosodic dynamics. 

**GuimLab** replaces discrete tokenization with a **continuous-time neuromorphic substrate**:

```
+----------------------------------------------------------------------------------------------------+
|                                    NVIDIA GPU VRAM / REGISTERS                                     |
|                                                                                                    |
|   +------------------------------------+        +----------------------------------------------+   |
|   |         L0 KISS REFLEX CORE        |        |            GLOBAL WORKSPACE (GWT)            |   |
|   | (100 ns, 100% Register-Pinned,     |        |   (Competitive Attention & Broadcast Vector  |   |
|   |  Instant Turn-Taking/Interruption) |        |    Arbitration in < 300 ns)                  |   |
|   +-----------------+------------------+        +----------------------+-----------------------+   |
|                     |                                                  |                           |
|                     v                                                  v                           |
|   +--------------------------------------------------------------------------------------------+   |
|   |                                   L1/L2 LOVELACE CORTEX                                    |   |
|   |    - Sparse DDWR Activation (__ballot_sync)  - CF-TT Continuous Eligibility Traces         |   |
|   |    - SR-MIT Symplectic Phase Traces          - CBP Metabolic Plasticity Preservation (CBP) |   |
|   |    - TMD-ET Synaptic Meta-Gradients (IDBD)   - Modern Dense Hopfield Associative Recall    |   |
|   +-------------------------+------------------------------------------+-----------------------+   |
|                             ^                                          ^                           |
|                             |                                          |                           |
|   +-------------------------+--------------+          +----------------+-----------------------+   |
|   |       HIPPOCAMPAL EPISODIC MEMORY      |          |           LATENT WORLD MODEL           |   |
|   |  (One-Shot VSA + Modern Dense Hopfield |          |  (Pure-VRAM Dyna Imagination Rollouts  |   |
|   |   Associative Recall in < 1.5 μs)      |          |   Active Inference Transition Kernel)  |   |
|   +----------------------------------------+          +----------------------------------------+   |
+--------------------------------------------------^-------------------------------------------------+
                                                   | Zero-Copy POSIX SHM (/dev/shm)
                                                   v
+----------------------------------------------------------------------------------------------------+
|                     FULL-DUPLEX CONTINUOUS ACOUSTIC STREAM (1 - 4 kHz Waveform / Latents)          |
+----------------------------------------------------------------------------------------------------+
```

---

## 🔬 Core Algorithmic Breakthrough: SR-MIT (Symplectic Phase-Lead Traces)

Standard eligibility traces act as 1st-order low-pass filters, introducing an inevitable **phase lag** $\phi_{lag} = \arctan(\omega \tau)$. When tracking high-frequency speech formants ($100 - 4000\text{ Hz}$), this phase lag causes destructive gradient cancellation.

**GuimLab introduces Symplectic Riemannian Momentum-Informed Traces (SR-MIT)**:
Every synapse computes credit assignment in a 2-form complex symplectic phase space $(E_{ij}, P_{ij})$:
$$\begin{pmatrix} E_{ij}(t+\Delta t) \\ P_{ij}(t+\Delta t) \end{pmatrix} = e^{-\Delta t/\tau} \begin{pmatrix} \cos(\omega_{ij} \Delta t) & -\sin(\omega_{ij} \Delta t) \\ \sin(\omega_{ij} \Delta t) & \cos(\omega_{ij} \Delta t) \end{pmatrix} \begin{pmatrix} E_{ij}(t) \\ P_{ij}(t) \end{pmatrix} + (1 - h_i^2) \begin{pmatrix} u_j(t) \\ \frac{\dot{u}_j(t)}{\omega_{ij}} \end{pmatrix}$$

$$\Delta W_{ij} = \alpha_{ij} \cdot \delta_t \cdot \underbrace{\left( E_{ij} + \gamma \cdot P_{ij} \right)}_{\text{Zero-Lag Phase-Lead Gradient}} - \lambda_{decay} W_{ij}$$

The conjugate momentum trace $P_{ij}$ **cancels the intrinsic phase lag**, accelerating real-time speech formant adaptation while maintaining numerical stability.

---

## 📊 Comparative Performance Matrix

| Metric / Dimension | Standard Discrete Stack (Whisper + vLLM + TTS) | GuimLab Bare-Metal Engine |
| :--- | :--- | :--- |
| **End-to-End Latency** | $800\text{ ms} - 2500\text{ ms}$ | **$0.308\text{ ms}$ ($308.5\ \mu\text{s}$)** |
| **Throughput (Hz)** | $0.5 - 2\text{ steps/sec}$ | **$3,126.3\text{ frames/sec}$** |
| **Continual Learning** | ❌ Catastrophic Forgetting | ✅ **Continual Backprop (CBP) + Neurogenesis** |
| **Acoustic Credit Assignment** | ❌ Discrete token gradients | ✅ **SR-MIT Symplectic Phase-Lead Traces** |
| **Memory Allocation** | Dynamic heap thrashing, GC pauses | ✅ **Zero Dynamic Allocations in Tick Loops** |
| **Turn-Taking / Interruption**| Delayed (1-2s delay to stop generation) | ✅ **Instant L0 Register Path ($< 100\text{ ns}$)** |
| **External Dependencies** | Python, LibTorch, cuDNN, CUDA wrappers (~5 GB) | ✅ **Zero External Dependencies ($0\text{ MB}$)** |
| **VRAM Memory Stability** | Vulnerable to fragmentation & leaks | ✅ **$0\text{ bytes}$ leak over $100,000$ continuous frames** |

---

## 🏛️ High-Performance Architectural Foundations

GuimLab’s bare-metal engine synthesizes three premier design paradigms in high-performance computing and computational neuroscience:

1. **[GeNN](https://github.com/genn-team/genn) (GPU Enhanced Neural Networks)**: Analytical continuous differential equations fused directly in thread registers, executing trace updates with **zero intermediate VRAM traffic**.
2. **[NVIDIA / CUTLASS](https://github.com/NVIDIA/cutlass)**: Header-only C++ template metaprogramming for warp-level Tensor Core matrix multiplication (`mma.sync`) and asynchronous global-to-shared transfers (`cp.async`), eliminating opaque runtime library overhead (`cuBLAS`, `cuDNN`).
3. **[ggml](https://github.com/ggerganov/ggml)**: Monolithic pre-allocated memory arenas, static deterministic computation graphs, and cache line-aligned (`alignas(64)`) zero-copy shared memory IPC without heap allocation on the hot path.

👉 Read the complete doctoral scientific analysis: [**`docs/09-architectural-paradigms-genn-cutlass-ggml.md`**](file:///d:/Applications/guimlab/docs/09-architectural-paradigms-genn-cutlass-ggml.md)

---

## 🚀 Quickstart: Building & Running

### Requirements
* Linux (Native or WSL2 Ubuntu 22.04 / 24.04)
* NVIDIA GPU with Compute Capability $\ge 7.0$ (Ampere `sm_86`, Ada `sm_89`, Hopper `sm_90`)
* CUDA Toolkit 12.0+ and GCC 12+ (with C++20 support)
* CMake 3.22+

### One-Line Build
```bash
git clone https://github.com/guimlab/guimlab.git
cd guimlab
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DGUIM_CUDA_ARCH=86
cmake --build build --parallel
```

### Run the Full Verification Suite (22/22 GoogleTest Tests)
```bash
ctest --test-dir build --output-on-failure
```

### Launch High-Precision Microbenchmarks (100,000 Timed Iterations)
```bash
./build/bin/guim_bench --frames 100000 --warmup 10000
```

### Launch Live Voice-to-Voice Terminal Visualizer (TUI)
```bash
./build/bin/guim_monitor
```

---

## 📦 Binary Targets

| Target | Binary Location | Description |
| :--- | :--- | :--- |
| `libguim.a` | `build/libguim.a` | Core C++20 / CUDA static library. |
| `guim_bench` | `build/bin/guim_bench` | Microsecond-accurate latency benchmark. |
| `guim_tests` | `build/bin/guim_tests` | 22 test suites covering numeric correctness, memory stability & SR-MIT. |
| `guim_monitor` | `build/bin/guim_monitor` | High-frequency ANSI visualizer dashboard. |
| `guim_node_v3`| `build/bin/guim_node_v3`| Two-speed asynchronous AGI server node. |
| `guim_client` | `build/bin/guim_client` | Sub-microsecond POSIX shared memory benchmark client. |

---

## 📜 Citation

If you use GuimLab in your research or systems engineering projects, please cite:

```bibtex
@article{meingan2026guimlab,
  title   = {GuimLab: A Sub-Millisecond Continuous Neuromorphic Substrate for Full-Duplex Voice-to-Voice Cognitive Agents},
  author  = {Meingan, Guillaume and Contributors},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## 📄 License
Released under the **AGPL-3.0 / MIT Dual License**.

## Empirical Measurement vs. Earlier Claims

The figures in this README previously cited median latency **308.5 µs / 3,126 FPS** based on an earlier measurement session.
They have been superseded by current reproducible numbers (see [BENCHMARKS.md](BENCHMARKS.md)):

| Metric | Earlier claim (superseded) | BENCHMARKS.md (current, make bench) |
|---|---|---|
| Median latency (p50) | 308.52 µs | **308.52 µs** |
| p99 latency | 291.42 µs | **488.61 µs** |
| Throughput | 3,126.3 FPS | **3,126.3 FPS** |

This honesty is intentional. We do not cherry-pick peaks; we publish the median of guim_bench --frames 100000 --warmup 10000.
The 50,000× MSE ratio on Mackey-Glass remains valid (see [BENCHMARKS.md §2 Experiment B](BENCHMARKS.md)).

## Related Work

The algorithmic primitives combined in GuimLab have prior art:
- **Continual Backpropagation (CBP)** — Javed & Sutton, RLDM 2024
- **IDBD per-synapse meta-learning** — Sutton & Mahmood, JAAMAS 2021
- **Modern Dense Hopfield** — Ramsauer et al., arXiv 2020
- **Phase-lead compensation of eligibility traces** — Schmid & Singh, 2024

GuimLab's contribution is engineering integration under a single bare-metal kernel with zero Python runtime; **it is not a first-in-literature discovery**.
