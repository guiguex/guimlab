# Native C++ GUI & HUD (Dear ImGui + ImPlot)

Guidelines for high-performance visual cockpit implementation in GuimLab Studio.

## 1. Ergonomic & Performance Rules
- **Pure C++20 & Zero Allocations**: The GUI rendering loop must never invoke dynamic heap allocations (`new`, `malloc`, `vector::resize`).
- **Static Circular Buffers**: All real-time telemetry curves, audio oscilloscopes, and latency plots must read from static pre-allocated ring buffers.
- **Deep-Tech Theme**: Maintain the Obsidian background (`#0A0D12`), Quantum Emerald (`#26E08C`), and Cyan (`#00D9E6`) visual hierarchy with ImGui Docking enabled.

## 2. Numeric Control Contracts
- Any interactive slider (such as Dopamine injection or Plasticity thresholds) must enforce:
  - Explicit bounds checking (`std::clamp`).
  - Lock-free transmission over IPC without stalling GPU kernel execution.
  - Clear visual feedback displaying live readbacks from the GPU.
