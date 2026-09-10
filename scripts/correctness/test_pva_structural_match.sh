#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUDIT_DIR="${1:-/tmp/pva_structural_match_test}"
RAISE_DIR="$AUDIT_DIR/raise"
DETECTOR="$REPO_ROOT/scripts/correctness/pva_structural_match.py"
PRODUCTION_MATCHER="$REPO_ROOT/scripts/correctness/kernel_match_rewrite.py"
PYTHON="${PYTHON:-/usr/bin/python3}"
mkdir -p "$AUDIT_DIR"

"$REPO_ROOT/scripts/correctness/audit_pva_semantic_fixtures.sh" "$RAISE_DIR" \
  >"$AUDIT_DIR/raising.log"

check_candidate() {
  local stem="$1" expected="$2"
  "$DETECTOR" "$RAISE_DIR/$stem.tensor.mlir" | grep -q "^$expected"
}

check_candidate box_filter3x3 BoxFilter3x3U8
check_candidate gaussian_filter3x3 GaussianFilter3x3U8
check_candidate bilateral_filter3x3 BilateralFilter3x3U8
check_candidate morphology_dilate3x3 MorphologyDilate3x3U8
check_candidate image_histogram_u8 ImageHistogramU8
check_candidate histogram_equalization HistogramEqualizationU8

# Recognition must not depend on the source function symbol.
sed 's/@fixture_box_filter3x3/@arbitrary_renamed_symbol/g' \
  "$RAISE_DIR/box_filter3x3.tensor.mlir" >"$AUDIT_DIR/renamed.mlir"
"$DETECTOR" "$AUDIT_DIR/renamed.mlir" | grep -q '^BoxFilter3x3U8'

# A structurally similar average with the wrong normalization is not box3x3.
sed -E 's/(arith\.constant )9( : i32)/\18\2/' \
  "$RAISE_DIR/box_filter3x3.tensor.mlir" >"$AUDIT_DIR/wrong_divisor.mlir"
if "$DETECTOR" "$AUDIT_DIR/wrong_divisor.mlir" | grep -q '^BoxFilter3x3U8'; then
  echo 'wrong-divisor negative test matched unexpectedly' >&2
  exit 1
fi

for stem in image_blend_u8 dl_activation_relu dl_softmax; do
  if [[ -n "$("$DETECTOR" "$RAISE_DIR/$stem.tensor.mlir")" ]]; then
    echo "unrelated negative fixture matched unexpectedly: $stem" >&2
    exit 1
  fi
done

# The production rewrite is intentionally stricter than candidate detection.
# Source-semantics-exact operations are executable selections by default.
production_match() {
  local stem="$1" expected="$2"
  "$PYTHON" "$PRODUCTION_MATCHER" "$RAISE_DIR/$stem.tensor.mlir" \
    --enable-structured-rewrite --dry-run 2>&1 | grep -q "$expected"
}
production_match morphology_dilate3x3 'pvaMorphologyDilate_3x3_u8'
production_match image_histogram_u8 'pvaImageHistogram_256_u8_u32'
production_match gaussian_filter3x3 'pvaGaussianFilter_3x3_u8'
production_match gaussian_filter3x3_s8 'pvaGaussianFilter_3x3_s8'
production_match gaussian_filter3x3_u16 'pvaGaussianFilter_3x3_u16'
production_match gaussian_filter3x3_s16 'pvaGaussianFilter_3x3_s16'
production_match image_histogram_u16 'pvaImageHistogram_256_u16_u32'
"$PYTHON" "$PRODUCTION_MATCHER" \
  "$RAISE_DIR/image_histogram_u8_s32.tensor.mlir" \
  --enable-structured-rewrite --pva-histogram-output-type s32 --dry-run \
  2>&1 | grep -q 'pvaImageHistogram_256_u8_s32'
"$PYTHON" "$PRODUCTION_MATCHER" \
  "$RAISE_DIR/image_histogram_u16_s32.tensor.mlir" \
  --enable-structured-rewrite --pva-histogram-output-type s32 --dry-run \
  2>&1 | grep -q 'pvaImageHistogram_256_u16_s32'
for stem in box_filter3x3 bilateral_filter3x3 histogram_equalization; do
  if "$PYTHON" "$PRODUCTION_MATCHER" "$RAISE_DIR/$stem.tensor.mlir" \
      --enable-structured-rewrite --dry-run 2>&1 | grep -q 'pva.*whole-image-operation'; then
    echo "non-exact PVA candidate selected unexpectedly: $stem" >&2
    exit 1
  fi
done

# Approximate numerical substitutions require an explicit, per-operation
# budget. The rewritten IR retains that policy for downstream auditing.
approximate_match() {
  local stem="$1" key="$2" budget="$3" expected="$4"
  local output="$AUDIT_DIR/$stem.approx.mlir"
  "$PYTHON" "$PRODUCTION_MATCHER" "$RAISE_DIR/$stem.tensor.mlir" \
    --enable-structured-rewrite \
    --pva-approximation-budget "$key=$budget" >"$output"
  grep -q "$expected" "$output"
  grep -q 'polygeist.numerical_contract = "approximate"' "$output"
  grep -q "polygeist.max_abs_error_budget = $budget : i64" "$output"
}
approximate_match box_filter3x3 box-filter 1 pvaBoxFilter_3x3_u8
approximate_match box_filter3x3_s8 box-filter 2 pvaBoxFilter_3x3_s8
approximate_match box_filter3x3_u16 box-filter 1 pvaBoxFilter_3x3_u16
approximate_match box_filter3x3_s16 box-filter 2 pvaBoxFilter_3x3_s16
approximate_match bilateral_filter3x3 bilateral-filter 2 pvaBilateralFilter_3x3_u8
approximate_match histogram_equalization histogram-equalization 1 \
  pvaHistogramEqualization_u8

# Opting into one approximate operation must not enable another.
if "$PYTHON" "$PRODUCTION_MATCHER" \
    "$RAISE_DIR/bilateral_filter3x3.tensor.mlir" \
    --enable-structured-rewrite \
    --pva-approximation-budget box-filter=2 2>&1 | \
    grep -q 'kernel.launch @pvaBilateralFilter'; then
  echo 'box approximation policy enabled bilateral unexpectedly' >&2
  exit 1
fi


# A symmetric-looking axis whose normalized/quantized weights do not prove the
# same operator must not be selected.
sed -E 's/(arith\.constant )2( : i32)/\13\2/' \
  "$RAISE_DIR/gaussian_filter3x3.tensor.mlir" \
  >"$AUDIT_DIR/non_equivalent_gaussian.mlir"
if "$PYTHON" "$PRODUCTION_MATCHER" "$AUDIT_DIR/non_equivalent_gaussian.mlir" \
    --enable-structured-rewrite --dry-run 2>&1 | \
    grep -q 'pvaGaussianFilter_3x3_u8'; then
  echo 'non-equivalent Gaussian quantization selected unexpectedly' >&2
  exit 1
fi

# Production selection is also invariant to the source symbol.
sed 's/@fixture_morphology_dilate3x3/@another_arbitrary_symbol/g' \
  "$RAISE_DIR/morphology_dilate3x3.tensor.mlir" \
  >"$AUDIT_DIR/renamed_morphology.mlir"
"$PYTHON" "$PRODUCTION_MATCHER" "$AUDIT_DIR/renamed_morphology.mlir" \
  --enable-structured-rewrite --dry-run 2>&1 | \
  grep -q 'pvaMorphologyDilate_3x3_u8'
sed 's/@fixture_gaussian_filter3x3/@renamed_gaussian_symbol/g' \
  "$RAISE_DIR/gaussian_filter3x3.tensor.mlir" \
  >"$AUDIT_DIR/renamed_gaussian.mlir"
"$PYTHON" "$PRODUCTION_MATCHER" "$AUDIT_DIR/renamed_gaussian.mlir" \
  --enable-structured-rewrite --dry-run 2>&1 | \
  grep -q 'pvaGaussianFilter_3x3_u8'
"$PYTHON" "$PRODUCTION_MATCHER" "$AUDIT_DIR/renamed.mlir" \
  --enable-structured-rewrite --dry-run 2>&1 | \
  grep -q '  match.*pvaBoxFilter' && {
    echo 'non-exact renamed box candidate selected unexpectedly' >&2
    exit 1
  }

echo 'PASS: typed exact defaults and explicit approximate routes; safety tests passed'
