#!/usr/bin/env bash
# Cross-build every raised-C Box, Gaussian, and ImageHistogram datatype route,
# then compare complete outputs on the configured Orin silicon profile.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/correctness/common_env.sh"

OUT_DIR="${1:-/tmp/pva_all_raised_dtypes}"
LOG="$OUT_DIR/campaign.log"
mkdir -p "$OUT_DIR"
: >"$LOG"

run_filter() {
  local operation="$1" suffix="$2" function="$3" ctype="$4"
  local symbol="$5" kind="$6" budget="$7" tolerance="$8"
  local exe="$OUT_DIR/${operation}_${suffix}"
  export POLYGEIST_ABI_BACKEND=pva
  export POLYGEIST_ONLY_KERNELS="$symbol"
  export POLYGEIST_PVA_HISTOGRAM_OUTPUT_TYPE=
  if [ "$budget" = exact ]; then
    export POLYGEIST_PVA_APPROXIMATION_BUDGETS=
  else
    export POLYGEIST_PVA_APPROXIMATION_BUDGETS="box-filter=$budget"
  fi
  export POLYGEIST_HARNESS_CFLAGS="-DPVA_FILTER_CTYPE=${ctype} -DPVA_FILTER_FUNCTION=${function} -DPVA_FILTER_DTYPE=${suffix} -DPVA_FILTER_KIND=${kind} -DPVA_FILTER_TOLERANCE=${tolerance}"
  "$REPO_ROOT/scripts/correctness/polygeist_build.sh" --target=jetson \
    --function="$function" \
    "$REPO_ROOT/test/Inputs/pva_matchers/image_processing.c" \
    --harness="$REPO_ROOT/scripts/correctness/pva_filter_matched_dtype_harness.c" \
    -o "$exe" 2>&1 | tee -a "$LOG"
  POLYGEIST_SILICON_PROFILE="${POLYGEIST_SILICON_PROFILE:-manual}" \
    "$REPO_ROOT/scripts/correctness/run_jetson.sh" --exe "$exe" \
    "pva_raised_${operation}_${suffix}" 2>&1 | tee -a "$LOG"
}

run_histogram() {
  local input_suffix="$1" output_suffix="$2" function="$3"
  local input_ctype="$4" output_ctype="$5" shift="$6" symbol="$7"
  local route="${input_suffix}_${output_suffix}"
  local exe="$OUT_DIR/histogram_${route}"
  export POLYGEIST_ABI_BACKEND=pva
  export POLYGEIST_ONLY_KERNELS="$symbol"
  export POLYGEIST_PVA_APPROXIMATION_BUDGETS=
  export POLYGEIST_PVA_HISTOGRAM_OUTPUT_TYPE="$output_suffix"
  export POLYGEIST_HARNESS_CFLAGS="-DPVA_HIST_INPUT_CTYPE=${input_ctype} -DPVA_HIST_OUTPUT_CTYPE=${output_ctype} -DPVA_HIST_FUNCTION=${function} -DPVA_HIST_ROUTE=${route} -DPVA_HIST_SHIFT=${shift}"
  "$REPO_ROOT/scripts/correctness/polygeist_build.sh" --target=jetson \
    --function="$function" \
    "$REPO_ROOT/test/Inputs/pva_matchers/image_processing.c" \
    --harness="$REPO_ROOT/scripts/correctness/pva_histogram_matched_dtype_harness.c" \
    -o "$exe" 2>&1 | tee -a "$LOG"
  POLYGEIST_SILICON_PROFILE="${POLYGEIST_SILICON_PROFILE:-manual}" \
    "$REPO_ROOT/scripts/correctness/run_jetson.sh" --exe "$exe" \
    "pva_raised_histogram_${route}" 2>&1 | tee -a "$LOG"
}

# These are explicit approximation contracts, validated below against complete
# outputs. They are not advertised as bit-exact substitutions.
run_filter box u8 fixture_box_filter3x3 uint8_t pvaBoxFilter_3x3_u8 1 1 1
run_filter box s8 fixture_box_filter3x3_s8 int8_t pvaBoxFilter_3x3_s8 1 2 2
run_filter box u16 fixture_box_filter3x3_u16 uint16_t pvaBoxFilter_3x3_u16 1 1 1
run_filter box s16 fixture_box_filter3x3_s16 int16_t pvaBoxFilter_3x3_s16 1 2 2

run_filter gaussian u8 fixture_gaussian_filter3x3 uint8_t pvaGaussianFilter_3x3_u8 2 exact 0
run_filter gaussian s8 fixture_gaussian_filter3x3_s8 int8_t pvaGaussianFilter_3x3_s8 2 exact 0
run_filter gaussian u16 fixture_gaussian_filter3x3_u16 uint16_t pvaGaussianFilter_3x3_u16 2 exact 0
run_filter gaussian s16 fixture_gaussian_filter3x3_s16 int16_t pvaGaussianFilter_3x3_s16 2 exact 0

run_histogram u8 u32 fixture_image_histogram_u8 uint8_t uint32_t 0 pvaImageHistogram_256_u8_u32
run_histogram u8 s32 fixture_image_histogram_u8_s32 uint8_t int32_t 0 pvaImageHistogram_256_u8_s32
run_histogram u16 u32 fixture_image_histogram_u16 uint16_t uint32_t 8 pvaImageHistogram_256_u16_u32
run_histogram u16 s32 fixture_image_histogram_u16_s32 uint16_t int32_t 8 pvaImageHistogram_256_u16_s32

{
  printf 'family,dtype,status,mismatches,max_error,tolerance_or_total\n'
  grep -E '^(FILTER|HIST)_RAISED,' "$LOG" | awk -F, '
    $1 == "FILTER_RAISED" {
      sub(/^mismatches=/, "", $5); sub(/^max_error=/, "", $6)
      sub(/^tolerance=/, "", $7)
      print $2 "," $3 "," $4 "," $5 "," $6 "," $7
    }
    $1 == "HIST_RAISED" {
      sub(/^mismatches=/, "", $4); sub(/^max_error=/, "", $5)
      sub(/^total=/, "", $6)
      print "histogram," $2 "," $3 "," $4 "," $5 "," $6
    }'
} >"$OUT_DIR/results.csv"
cat "$OUT_DIR/results.csv"
