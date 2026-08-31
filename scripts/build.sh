#!/usr/bin/env bash
# ============================================================================
#  scripts/build.sh — configure + build + benchmark Guimlab
#  ============================================================================
#  Usage: bash scripts/build.sh [Release|Debug|RelWithDebInfo]
#
#  Workflow:
#   1. Check GPU
#   2. Configure CMake (fresh build/ dir)
#   3. Build all targets (parallel)
#   4. Run benchmark if Release build
# ============================================================================

set -euo pipefail

BUILD_TYPE="${1:-Release}"
BUILD_DIR="build"

echo "============================================================"
echo "  Guimlab — build ($BUILD_TYPE)"
echo "============================================================"

# Step 1: check GPU
bash "$(dirname "$0")/check_gpu.sh"

# Step 2: configure CMake
echo ""
echo "[1/3] Configuring CMake..."
cmake -B "$BUILD_DIR" -S . \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Step 3: build
echo ""
echo "[2/3] Building..."
cmake --build "$BUILD_DIR" --parallel "$(nproc 2>/dev/null || echo 4)"

# Step 4: run benchmark (Release only)
if [ "$BUILD_TYPE" = "Release" ]; then
    echo ""
    echo "[3/3] Running benchmark..."
    if [ -f "$BUILD_DIR/bin/guim_bench" ]; then
        "$BUILD_DIR/bin/guim_bench"
    else
        echo "WARN: guim_bench not found (may not be built yet)."
    fi
fi

echo ""
echo "[OK] Build complete. Output in $BUILD_DIR/bin/"