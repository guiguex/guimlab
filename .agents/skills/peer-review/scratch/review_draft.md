# Peer-review working draft — REVIEW-GUIMLAB-001

> Private working document. Human review, policy checks, and factual verification are required. Do not submit this scaffold with unresolved placeholders. Do not make or announce an editorial decision.

## Intake record

- Reviewer capacity: `author_requested_reader`
- Peer-review model: `open`
- Declared processing plan: `local_deterministic_tools`
- Manuscript text is not embedded by the generator.
- Reconfirm conflicts, competence limits, tool use, confidentiality, and deletion or retention obligations before submission.

# Comments to authors

## Evidence-bounded summary

The manuscript presents GuimLab, a native C++20 and pure CUDA neuromorphic runtime substrate designed for sub-millisecond continuous-time recurrent processing in full-duplex conversational voice agents. The core architecture replaces discrete tokenized pipelines (ASR-LLM-TTS) with a continuous-time dynamical system governed by five primary mechanisms: (1) Symplectic Riemannian Momentum-Informed Traces (SR-MIT) to mitigate harmonic phase lag in acoustic credit assignment; (2) Closed-Form Continuous-Time Eligibility Traces (CF-TT); (3) Temporal Meta-Descent (TMD-ET / IDBD) for per-synapse adaptive learning rates; (4) Continual Backpropagation (CBP) with running activation variance tracking and asynchronous GPU neurogenesis; and (5) a Two-Speed hierarchy comprising an L0 spinal reflex core and an L1/L2 sparse cortex with Global Workspace arbitration and Modern Dense Hopfield episodic memory. Empirical benchmarks on an NVIDIA RTX 3090 report a median round-trip latency of 308.5 microseconds, sustained throughput of 3,126 frames/second, 0 bytes of dynamic allocation on hot paths, and 0 bytes of memory leakage over 100,000 frames.

## Strengths

1. **Bare-Metal Engineering Integrity**: The engine is implemented entirely in native C++20 and pure CUDA with zero external runtime dependencies (no Python, PyTorch, LibTorch, or cuDNN in the critical path). Memory allocation follows static arenas and lock-free ring buffers, yielding 0 bytes of dynamic allocation and zero memory leaks over 100,000 frames.
2. **Symplectic Phase-Space Formulation for Traces**: Formulating eligibility traces as a complex unitary rotation in phase space $(E_{ij}, P_{ij})$ addresses the fundamental harmonic phase delay $\arctan(\omega \tau)$ inherent in first-order low-pass eligibility traces when tracking oscillatory acoustic signals.
3. **Rigorous PTX Register Analysis**: The codebase demonstrates meticulous GPU resource management, maintaining zero register spilling (0 bytes stack frame, 0 spill stores/loads) across all primary compute kernels (`kiss_reflex_persistent_kernel`, `symplectic_step_kernel`, `guim_core_step_kernel`).
4. **Comprehensive Local Test Suite**: The repository features 23 green test suites covering functional correctness, numerical bounds under float32, plasticity monotonicity, and chaotic attractor synchronization.

## Major comments

### Major comment M1

- Location: Section 2.2, Equations (37)-(41) in LaTeX; `src/symplectic_traces_kernel.cu:L150-L154`
- Observation: Equation (41) expresses the synaptic weight update with an empirical scalar parameter $\gamma$: $\Delta W_{ij} = \alpha_{ij} \delta(t) (E_{ij}(t) + \gamma P_{ij}(t)) - \lambda_{decay} W_{ij}$. However, in the CUDA implementation, the phase-lead scaling coefficient is derived analytically from the Kuramoto eigenfrequency $\omega_{ij}$ and time constant $\tau$ via $\gamma(\omega_{ij}, \tau) = \frac{\omega_{ij} \tau}{\sqrt{1 + (\omega_{ij} \tau)^2}} = \sin(\arctan(\omega_{ij} \tau))$.
- Evidence or criterion: Equation (41) in `paper/guimlab_whitepaper.tex` vs `src/symplectic_traces_kernel.cu` lines 151-153.
- Why it matters: The analytical derivation of $\gamma(\omega_{ij}, \tau)$ represents a significant theoretical strength that connects directly to the phase lag $\phi_{lag} = \arctan(\omega \tau)$. Describing it only as an unconstrained parameter $\gamma$ in the text diminishes the mathematical elegance and physical grounding of the contribution.
- Requested action: Update Section 2.2 to state the analytical closed-form expression for $\gamma_{ij}(t)$ as a function of $\omega_{ij} \tau$ and provide the physical intuition connecting it to exact phase-lead compensation.

### Major comment M2

- Location: Abstract, Section 1, Section 4.2; `tests/test_symplectic_discovery.cpp`
- Observation: The manuscript frames GuimLab as a full-duplex voice-to-voice substrate for continuous dialogue. However, the empirical convergence benchmarks presented in Section 4.2 and tested in `test_symplectic_discovery.cpp` evaluate synthetic multi-harmonic signals (220 Hz formants) and the Mackey-Glass chaotic delay differential attractor, rather than real continuous acoustic latents (such as EnCodec or DAC frame embeddings).
- Evidence or criterion: Benchmark implementations in `tests/test_symplectic_discovery.cpp` and `BENCHMARKS.md:L32-L48`.
- Why it matters: While synthetic harmonic tracking and Mackey-Glass chaos convincingly validate phase-space convergence, speech acoustics involve non-stationary broadband dynamics, multi-talker variance, and background acoustic noise. Clarifying the empirical scope prevents overgeneralization of conversational capabilities.
- Requested action: Clarify in Section 4 that the present benchmarks demonstrate foundational mathematical convergence on synthetic multi-harmonic and chaotic dynamics, and explicitly discuss future evaluation on neural audio codebook streams in Section 6.

### Major comment M3

- Location: Section 3, paragraph 1; `src/kiss_reflex_kernel.cu:L34-L36`
- Observation: Section 3 states that the L0 reflex core "allocates state vectors and synaptic weights directly into GPU thread registers (`__launch_bounds__(32, 1)`)". However, the CUDA kernel `kiss_reflex_persistent_kernel` allocates these weights in on-chip Shared Memory (`__shared__ float s_weights[16][128]`, totaling 16,960 bytes of SRAM), using thread registers for per-thread accumulators and loop variables.
- Evidence or criterion: Shared memory declarations in `src/kiss_reflex_kernel.cu` lines 34-36 and compilation statistics in `BENCHMARKS.md:L56`.
- Why it matters: Shared Memory (SRAM) and thread private registers are distinct architectural hardware hierarchies in NVIDIA GPUs with different allocation limits and latency profiles. Precise systems terminology is vital for reproducibility and architectural clarity.
- Requested action: Refine Section 3.1 to state that synaptic state and weights reside in on-chip GPU Shared Memory (SRAM) with zero global VRAM transactions per tick, executed by a single register-pinned warp.

### Major comment M4

- Location: Section 4.1, Table 1; `tests/bench_main.cpp`
- Observation: Table 1 reports a median host-to-device round-trip latency of 308.5 microseconds on an RTX 3090, but does not provide the latency decomposition separating PCIe host-to-device transfer, bare-metal kernel execution on Streaming Multiprocessors (SMs), device-to-host transfer, and IPC synchronization.
- Evidence or criterion: Benchmark timing harness in `tests/bench_main.cpp` and `BENCHMARKS.md:L13-L23`.
- Why it matters: On discrete GPUs, PCIe DMA transfers and driver launch overhead account for the majority of the ~300 microsecond round-trip time, whereas individual CUDA kernel execution times are on the order of single microseconds. Disaggregating these components highlights the true compute density of the bare-metal kernel.
- Requested action: Add a short breakdown in Section 4.1 specifying the isolated GPU kernel execution time versus the full host-to-device PCIe round-trip time.

## Minor comments

### Minor comment m1

- Location: Section 4.2; Table 1 in `paper/guimlab_whitepaper.tex` vs `BENCHMARKS.md:L65`
- Observation: Table 1 lists "Unit Tests Passed: 22 / 22 (100%)", whereas `BENCHMARKS.md` documents "Total Test Suites: 23 / 23".
- Evidence or criterion: Table 1 in `paper/guimlab_whitepaper.tex` and Section 4 in `BENCHMARKS.md`.
- Why it matters: Inconsistent test suite counts between the whitepaper and repository documentation can cause confusion during review.
- Requested action: Reconcile Table 1 to state 23 / 23 test suites to match the repository test harness.

### Minor comment m2

- Location: Section 2.5, Equation (50) in LaTeX / Section 2.5 in Markdown
- Observation: Equation (50) expresses Modern Hopfield retrieval using the summation index $m = 1 \dots M$ over memory slots ($\mathbf{k}_m, \mathbf{v}_m$), but the accompanying text describes complexity as "$O(N \cdot D)$ per query where $N = \text{number of stored entries}$".
- Evidence or criterion: Equation (50) and text in Section 2.5 of `paper/guimlab_whitepaper.md`.
- Why it matters: Using $M$ in the mathematical formula and $N$ in the textual complexity description is a minor notation inconsistency.
- Requested action: Use a single consistent variable name (either $M$ or $N$) for the memory capacity across Section 2.5.

### Minor comment m3

- Location: Section 5 in `paper/guimlab_whitepaper.tex`
- Observation: The LaTeX document presents citations in narrative form within the text (e.g., `(Javed \& Sutton, RLDM 2024)`) without formal `\cite{...}` commands or a standard bibliography environment.
- Evidence or criterion: Section 5 in `paper/guimlab_whitepaper.tex` lines 82-84.
- Why it matters: Scientific publications require complete bibliographic metadata (authors, venue, year, DOIs) formatted via standard LaTeX bibliography tools to enable citation indexing.
- Requested action: Introduce a formal BibTeX bibliography or `thebibliography` environment with complete reference entries for cited works (Dohare & Sutton 2024, Sutton & Mahmood 2021, Ramsauer et al. 2020, Schmid & Singh 2024).

### Minor comment m4

- Location: Section 2.1 and Section 2.2
- Observation: The continuous-time differential equation $\tau \frac{d\mathbf{h}(t)}{dt} = -\mathbf{h}(t) + \tanh(\dots)$ is stated in continuous physical time, but the nominal time-step $\Delta t$ used in discrete execution (e.g., $\Delta t = 0.5\text{ ms}$ for 2 kHz or $\Delta t = 20\text{ ms}$ for 50 Hz frame rates) and the integration constant $\tau$ are not explicitly specified in the text.
- Evidence or criterion: Section 2.1 vs `src/symplectic_traces_kernel.cu:L95-L98`.
- Why it matters: Providing standard operating parameters for $\Delta t$ and $\tau$ helps practitioners understand the time constant of continuous trace decay relative to speech frame rates.
- Requested action: Include a brief sentence in Section 2 specifying typical numerical values for $\Delta t$ and $\tau$ used in acoustic streaming.

## Methods, statistics, and reproducibility

The methodology is supported by complete C++20 and CUDA source implementations. Numerical stability checks in float32 arithmetic are verified across 23 automated test suites. Empirical timing microbenchmarks are based on 100,000 continuous timed iterations preceded by 10,000 warmup iterations on an RTX 3090 GPU. The memory arena design and lock-free ring buffers guarantee deterministic memory bounds. To further elevate methodological rigor, future iterations should include:
1. Multi-seed statistical dispersion (confidence intervals or standard deviations across independent initializations) for the convergence error rates in Experiment A and Experiment B.
2. Direct benchmarking against standard batched baseline implementations on representative neural audio codebooks.

## Ethics, transparency, figures, tables, and citations

1. **Transparency & Open Source**: The codebase is fully inspectable with clear build scripts (CMake), automated tests (GoogleTest), and benchmarking tools.
2. **Citations & Attributions**: Section 5 explicitly and appropriately acknowledges prior art for individual algorithmic components (Continual Backpropagation, IDBD, Modern Hopfield, phase-lead compensation), clearly positioning GuimLab's core contribution as bare-metal integration under a unified continuous substrate.

## Limitations of this review

This review was conducted locally by inspecting the source code, kernel implementations, build configuration, automated test suites, and empirical benchmarking reports within the repository. The timing benchmarks were validated against documented physical hardware execution logs (RTX 3090, CUDA 12.8, sm_86). No external network requests or closed third-party services were utilized.

# Confidential comments to editor

## Reviewer disclosures

- Conflicts and editor clearance: None identified; author-requested technical review.
- Competence limits or specialist review needed: Specialized in C++/CUDA high-performance computing, continuous-time recurrent networks, and speech processing. No specialist review required for the evaluated software substrate.
- Assistance or tools used and required disclosure: Local deterministic linting and structural analysis scripts from the local peer-review skill.
- Confidentiality or retention issue: Local processing only; no confidential text transmitted externally.

## Editorial-process or integrity concerns

No ethical, reproducibility, or integrity concerns identified. The mathematical formulations are reflected in the underlying CUDA kernel implementations, and claimed benchmark metrics are consistent with hardware profiling data.
