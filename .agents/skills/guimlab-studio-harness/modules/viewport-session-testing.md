# Viewport Session Testing & Continuous Replay

Protocol for automated regression testing of continuous neuromorphic streaming and visual feedback.

## 1. Full-Frame Invariance Gate
- Any change to the streaming audio pipeline, DDWR sparse filter, or sensory mapping must prove that unedited sensor channels remain bit-exact.
- No silent zero-masking or uninitialized buffers are accepted as valid tests.

## 2. Continuous Replay Pipeline
- **Audio Streaming Test**: Inject $10^5$ continuous acoustic frames via `stream_wav_demo` and assert:
  - Zero dropped frames in the lock-free ring buffer.
  - Zero memory leaks in VRAM ($0.00\,\text{B}$).
  - Convergence of the eligibility traces and continuous plasticity adaptation without numerical NaN/Inf divergence.
