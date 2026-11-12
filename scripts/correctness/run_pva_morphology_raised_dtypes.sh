#!/usr/bin/env bash
# Cross-build S8/U16/S16 from ordinary C through the structural matcher, then
# run complete-output comparisons on the configured Orin silicon profile.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-/tmp/pva_morphology_raised_dtypes}"
LOG="$OUT_DIR/campaign.log"
mkdir -p "$OUT_DIR"
: >"$LOG"

run_one() {
  local suffix="$1" function="$2" ctype="$3" minimum="$4"
  local exe="$OUT_DIR/morphology_${suffix}"
  export POLYGEIST_ABI_BACKEND=pva
  export POLYGEIST_ONLY_KERNELS="pvaMorphologyDilate_3x3_${suffix}"
  export POLYGEIST_HARNESS_CFLAGS="-DPVA_MORPH_CTYPE=${ctype} -DPVA_MORPH_FUNCTION=${function} -DPVA_MORPH_MIN=${minimum} -DPVA_MORPH_DTYPE=${suffix}"
  "$REPO_ROOT/scripts/correctness/polygeist_build.sh" --target=jetson \
    --function="$function" \
    "$REPO_ROOT/test/Inputs/pva_matchers/image_processing.c" \
    --harness="$REPO_ROOT/scripts/correctness/pva_morphology_matched_dtype_harness.c" \
    -o "$exe" 2>&1 | tee -a "$LOG"
  POLYGEIST_SILICON_PROFILE="${POLYGEIST_SILICON_PROFILE:-manual}" \
    "$REPO_ROOT/scripts/correctness/run_jetson.sh" --exe "$exe" \
    "pva_morphology_raised_${suffix}" 2>&1 | tee -a "$LOG"
}

run_one s8 fixture_morphology_dilate3x3_s8 int8_t INT8_MIN
run_one u16 fixture_morphology_dilate3x3_u16 uint16_t 0
run_one s16 fixture_morphology_dilate3x3_s16 int16_t INT16_MIN

grep '^MORPH_RAISED,' "$LOG" | tee "$OUT_DIR/results.csv"
