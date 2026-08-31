#!/usr/bin/env bash
# ============================================================================
#  scripts/check_l0_spill.sh — CI guard against L1 register spill in L0 kernels
#  ---------------------------------------------------------------------------
#  Parses the output of `nvcc -Xptxas -v` (enabled in CMakeLists.txt) for the
#  compiled kernel binaries in build/bin.  Fails the build if any kernel
#  exceeds the 255-register-per-thread cap, which would cause nvcc to silently
#  spill to L1 (destroying the sub-microsecond latency target).
#
#  Usage:
#    bash scripts/check_l0_spill.sh       # uses default build/ dir
#    bash scripts/check_l0_spill.sh path  # uses custom build dir
#
#  Exit codes:
#    0  — all kernels OK (used registers <= 200 to leave room for future features)
#    1  — build dir not found, or .cu file(s) not compiled yet
#    2  — one or more kernels exceed the threshold
# ============================================================================

set -euo pipefail

BUILD_DIR="${1:-build}"
THRESHOLD=200  # hard ceiling: warn at 200, fail at 255

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}FATAL:${NC} build dir '$BUILD_DIR' not found."
    echo "  Run 'cmake -B build -S . && cmake --build build' first."
    exit 1
fi

# Build log files are emitted by nvcc when -Xptxas -v is set.
# They live next to the .o files, with names like <kernel>.cu.ptxas
echo "================================================================"
echo "  L0 register usage audit (cap: 255 regs/thread)"
echo "================================================================"
echo

WARNINGS=0
ERRORS=0
declare -a KERNEL_REPORTS=()

# Iterate over all compiled kernel ptxas dumps
while IFS= read -r -d '' ptxas_file; do
    # Extract the registers used line for the main kernel
    # Format: "Used 32 registers, 96 bytes smem, 320 bytes lmem"
    used_regs=$(grep -oP 'Used \K[0-9]+(?= registers)' "$ptxas_file" | head -1 || echo "0")

    if [ -z "$used_regs" ]; then
        continue
    fi

    # Kernel name = parent file
    kernel_name=$(basename "$ptxas_file" .cu.ptxas)

    # Status color
    if [ "$used_regs" -ge 255 ]; then
        status="${RED}SPILL${NC}"
        ERRORS=$((ERRORS+1))
    elif [ "$used_regs" -ge "$THRESHOLD" ]; then
        status="${YELLOW}WARN${NC}"
        WARNINGS=$((WARNINGS+1))
    else
        status="${GREEN}OK${NC}"
    fi

    KERNEL_REPORTS+=("$(printf '%-32s %4d regs  %s' "$kernel_name" "$used_regs" "$status")")
done < <(find "$BUILD_DIR" -name "*.cu.ptxas" -print0 2>/dev/null || true)

if [ ${#KERNEL_REPORTS[@]} -eq 0 ]; then
    echo -e "${YELLOW}WARN:${NC} no .cu.ptxas files found in '$BUILD_DIR'."
    echo "  Did you build with -Xptxas -v (enabled in CMakeLists.txt)?"
    exit 1
fi

# Print sorted report
printf '%s\n' "${KERNEL_REPORTS[@]}" | sort

echo
echo "----------------------------------------------------------------"
echo "Threshold: $ THRESHOLD regs (warn),  255 regs (fail)"
echo -e "Result: ${WARNINGS} warning(s), ${ERRORS} error(s)"
echo "----------------------------------------------------------------"

if [ "$ERRORS" -gt 0 ]; then
    echo
    echo -e "${RED}BUILD FAILED${NC} — at least one kernel exceeds the 255-reg L0 cap."
    echo "Fix: reduce local array sizes, fuse computations, or split the kernel."
    exit 2
fi

if [ "$WARNINGS" -gt 0 ]; then
    echo
    echo -e "${YELLOW}CAUTION${NC} — at least one kernel is close to the cap."
    echo "Future features may push it over the edge."
fi

echo
echo -e "${GREEN}L0 register usage OK.${NC}"
exit 0