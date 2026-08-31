# Viewport Session Testing & Continuous Replay

Protocol for automated capture, recording, and replay of interactive visual sessions in GuimLab Studio.

## 1. Overview
Viewport session testing enables headless and GUI regression testing by recording:
- Input acoustic formants and injected neuromodulators.
- Exact frame-by-frame cortical states $h_i(t)$.
- Time-synchronized latency and FPS metrics.

## 2. Programmatic Session Replay Interface

```cpp
#include "guim_viewport_session.h"

// Initialize recorder
guim::studio::ViewportSessionRecorder recorder("session_001.json");
recorder.start();

// Record frame snapshot
recorder.record_frame(telemetry);

// Save and evaluate session
recorder.stop();
```

## 3. Automated Headless Verification
In CI/CD or agent development:
```bash
# Run headless session test
./build/bin/guim_tests --gtest_filter="StudioHarnessTest.*"
```
