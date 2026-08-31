# GuimLab Master Donor Reference Library

This source map lists the canonical reference implementations and algorithms curated to guide native C++20 and CUDA engineering in GuimLab without introducing external dependencies.

---

## 1. High-Performance GPU & Tensor Compute

| Reference | Category | License Tier | Key Signal & Usage |
| :--- | :--- | :--- | :--- |
| **NVIDIA CUTLASS** | Tensor Cores / MMA | Apache-2.0 | Reference for CUDA C++ template abstractions for high-performance matrix-multiplication on Tensor Cores (`mma.sync`). |
| **NVIDIA vk_mini_samples** | Modern Vulkan / Compute | Apache-2.0 | Clean, minimal modern Vulkan samples for compute shaders, descriptors, and fast offscreen pipelines. |
| **Khronos Vulkan-Samples** | Vulkan Foundations | Apache-2.0 | Reference for memory synchronization, timeline semaphores, and headless execution. |
| **FlashAttention (C++/CUDA)** | Exact Attention / Tiling | BSD-3-Clause | Memory-efficient tiling algorithms, online softmax integration, and register-level reduction. |
| **GGML / llama.cpp** | Bare-Metal Quantization | MIT | Reference for INT8/INT4 SIMD/CUDA quantization, AVX2 dequantization routines, and zero-allocation execution. |
| **Whisper.cpp** | Acoustic Processing | MIT | Reference for streaming acoustic audio ingestion and continuous mel-filterbank extraction in pure C++. |

---

## 2. Real-Time UI, HUD & Plotting

| Reference | Category | License Tier | Key Signal & Usage |
| :--- | :--- | :--- | :--- |
| **Dear ImGui** | Immediate-Mode HUD | MIT | Industry-standard immediate-mode GUI for tools, docking panels, and zero-overhead debug overlays. |
| **ImPlot** | GPU Real-Time Plotting | MIT | High-frequency plotting library for real-time oscilloscopes, heatmaps, phase-space trajectories, and histograms. |
| **ImGuizmo** | Spatial Controls | MIT | 3D transform gizmos and coordinate manipulation controls. |
| **RmlUi** | HTML/CSS GUI | MIT | Lightweight C++ UI library for specialized HUD layouts. |
| **Nuklear** | Single-Header GUI | MIT / Public Domain | Minimal immediate-mode C GUI for embedded or ultra-low footprint environments. |

---

## 3. Geometry, Simulation & Audio DSP

| Reference | Category | License Tier | Key Signal & Usage |
| :--- | :--- | :--- | :--- |
| **madmann91/bvh** | Standalone BVH | MIT | High-performance C++20 BVH construction and spatial queries. |
| **meshoptimizer** | Mesh & Memory Optimization | MIT | Cache-efficient memory layouts and vertex buffer indexing. |
| **Jolt Physics** | Rigid Body Simulation | MIT | Multi-threaded modern C++ physics and lock-free spatial trees. |
| **miniaudio** | Single-Header Audio | MIT / Public Domain | Low-latency audio playback and capture for real-time 16 kHz stream testing. |
| **libsoundio** | Low-Latency Audio I/O | MIT | Robust audio backend abstraction for Linux ALSA/PulseAudio/JACK and Windows WASAPI. |
