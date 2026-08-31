#!/usr/bin/env bash
# ============================================================================
#  scripts/check_gpu.sh — verify CUDA installation and GPU detection
#  ============================================================================
#  Exit 0 if a usable GPU is found, exit 1 otherwise.
#  Prints diagnostic info to stdout.
# ============================================================================

set -euo pipefail

echo "============================================================"
echo "  Guimlab — CUDA/GPU environment check"
echo "============================================================"

# ------------------ nvcc available? ------------------
if ! command -v nvcc >/dev/null 2>&1; then
    echo "FATAL: nvcc not found in PATH."
    echo "Install CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-toolkit"
    exit 1
fi

NVCC_VER=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',' | head -1)
echo "nvcc version:        $NVCC_VER"

# ------------------ nvidia-smi available? ------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: nvidia-smi not found."
    echo "Install NVIDIA driver matching CUDA Toolkit."
    exit 1
fi

# ------------------ Driver version ------------------
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "Driver version:      ${DRIVER_VER:-unknown}"

# ------------------ GPU model + compute capability ------------------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
SM_COUNT=$(nvidia-smi --query-gpu=multiprocessor_count --format=csv,noheader 2>/dev/null | head -1)
MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)

echo "GPU:                 ${GPU_NAME:-unknown}"
echo "Compute capability:  ${COMPUTE_CAP:-unknown}"
echo "SM count:            ${SM_COUNT:-unknown}"
echo "Total memory:        ${MEMORY_MB:-unknown}"

# ------------------ Compute capability check ------------------
# We require >= 7.0 (Volta, 2017). RTX 30xx/40xx/50xx are 8.6/8.9/9.0.
if [ -n "$COMPUTE_CAP" ]; then
    MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)
    if [ "$MAJOR" -lt 7 ]; then
        echo ""
        echo "WARN: Compute capability < 7.0 (Volta+)."
        echo "      Performance may be suboptimal; some patterns assume modern archs."
        echo "      Consider upgrading your GPU."
    fi
fi

echo ""
echo "[OK] CUDA environment looks usable."
exit 0