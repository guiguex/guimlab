# Supervisor Review Cadence & Adversarial Auditing

Guidelines for long-running, multi-slice development and multi-agent coordination.

## 1. Adversarial Review Cadence
- After every **3 implementation slices**, trigger an adversarial verification pass checking:
  1. No violation of the 4 constitutional prohibitions.
  2. No degradation of median latency or SM occupancy.
  3. Strict conformity with the C++20 / CUDA bare-metal standard.
- Unresolved review concerns act as hard blockers before committing or closing out features.

## 2. Slice Phase Telemetry
For complex kernel or architecture refactoring, log phase timing:
```text
[PHASE] Research: 15s | Implementation: 45s | Native Build: 12s | CTest Validation: 8s
```
