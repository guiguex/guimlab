#!/usr/bin/env bash
# =============================================================================
#  Guimlab  -  run benchmark executable
#  Builds if needed, then runs guim_bench.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${APP_DIR}/build"
EXE="${BUILD_DIR}/guim_bench"

if [[ ! -x "${EXE}" ]]; then
    echo "guim_bench not found - invoking build.sh first"
    "${SCRIPT_DIR}/build.sh" "$@"
fi

# Multi-config generators (VS) emit Release/guim_bench.exe
if [[ ! -x "${EXE}" && -x "${BUILD_DIR}/Release/guim_bench.exe" ]]; then
    EXE="${BUILD_DIR}/Release/guim_bench.exe"
fi

echo "==> running ${EXE}"
echo
exec "${EXE}" "$@"