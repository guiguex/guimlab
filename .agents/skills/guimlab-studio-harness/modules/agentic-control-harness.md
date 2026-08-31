# Agentic Control Harness

Guidelines for enabling autonomous AI agents to launch, control, test, and troubleshoot GuimLab.

## 1. Non-Interactive Driving
- Agents should launch headless verification modes using command-line arguments and test targets:
  ```bash
  # Run non-interactive studio harness verification
  ./build/bin/guim_tests --gtest_filter="StudioHarnessTest.*"
  ```
- All telemetry structures must support programmatic query and export (JSON/CSV) for automated non-regression evaluation.

## 2. Telemetry and Discovery Before Modification
- Before editing any IPC structure or engine parameters, inspect existing layouts (`include/ipc_structs_v2.h`, `include/ipc_structs_sparse.h`).
- Record baseline performance before modifying GPU kernels and verify with the Proof Contract table.
