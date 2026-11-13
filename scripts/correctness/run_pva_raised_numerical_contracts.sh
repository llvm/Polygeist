#!/usr/bin/env bash
# Cross-build and run the three structurally recognized PVA image operations
# whose raised-C arithmetic is not bit-identical to the vendor contract.
set -euo pipefail

CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CORRECTNESS_DIR/common_env.sh"

OUT_DIR="${1:-/tmp/pva_raised_numerical_contracts}"
LOG="$OUT_DIR/campaign.log"
RESULTS="$OUT_DIR/results.csv"
mkdir -p "$OUT_DIR"
: >"$LOG"

run_one() {
  local id="$1" function="$2" symbol="$3" operation="$4" budget="$5" smoke_op="$6"
  local exe="$OUT_DIR/$id"
  export POLYGEIST_ABI_BACKEND=pva
  export POLYGEIST_ONLY_KERNELS="$symbol"
  export POLYGEIST_PVA_APPROXIMATION_BUDGETS="$operation=$budget"
  export POLYGEIST_HARNESS_CFLAGS="-DPVA_SMOKE_OP=$smoke_op -DPVA_SMOKE_TOLERANCE=$budget"
  "$REPO_ROOT/scripts/correctness/polygeist_build.sh" --target=jetson \
    --function="$function" \
    "$REPO_ROOT/test/Inputs/pva_matchers/image_processing.c" \
    --harness="$REPO_ROOT/scripts/correctness/pva_matched_smoke_harness.c" \
    -o "$exe" 2>&1 | tee -a "$LOG"
  POLYGEIST_SILICON_PROFILE="${POLYGEIST_SILICON_PROFILE:-manual}" \
    "$REPO_ROOT/scripts/correctness/run_jetson.sh" --exe "$exe" \
    "pva_raised_${id}" 2>&1 | tee -a "$LOG"
}

run_one box_u8 fixture_box_filter3x3 pvaBoxFilter_3x3_u8 box-filter 1 1
run_one bilateral_u8 fixture_bilateral_filter3x3 \
  pvaBilateralFilter_3x3_u8 bilateral-filter 2 4
run_one histogram_equalization_u8 fixture_histogram_equalization \
  pvaHistogramEqualization_u8 histogram-equalization 1 6

printf 'operation,status,mismatches,max_error,tolerance,checksum\n' >"$RESULTS"
grep '^MATCHED_SMOKE,' "$LOG" | awk -F, '
  {
    op = $2 == "1" ? "box-u8" : ($2 == "4" ? "bilateral-u8" : "histogram-equalization-u8")
    sub(/^mismatches=/, "", $4)
    sub(/^max_error=/, "", $5)
    sub(/^tolerance=/, "", $6)
    sub(/^checksum=/, "", $7)
    print op "," $3 "," $4 "," $5 "," $6 "," $7
  }' >>"$RESULTS"
cat "$RESULTS"
