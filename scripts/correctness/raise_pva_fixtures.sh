#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

INPUT_DIR="$REPO_ROOT/test/Inputs/pva_matchers"
OUT_DIR="${1:-/tmp/pva_matcher_raise}"
mkdir -p "$OUT_DIR"

native_sources=(
  "$INPUT_DIR/pva_fixture_self_test.c"
  "$INPUT_DIR/conv3x3_i8.c"
  "$INPUT_DIR/box_filter_3x3_i8.c"
  "$INPUT_DIR/gaussian_filter_3x3_u8.c"
  "$INPUT_DIR/bilateral_filter_3x3_u8.c"
  "$INPUT_DIR/histogram_equalization_u8.c"
)
cc -O2 -std=c11 -Wall -Wextra -Werror "${native_sources[@]}" -lm \
  -o "$OUT_DIR/pva_fixture_self_test"
"$OUT_DIR/pva_fixture_self_test" | tee "$OUT_DIR/native_self_test.log"

fixtures=(
  "conv3x3_i8:pva_fixture_conv3x3_i8"
  "box_filter_3x3_i8:pva_fixture_box_filter_3x3_i8"
  "gaussian_filter_3x3_u8:pva_fixture_gaussian_filter_3x3_u8"
  "bilateral_filter_3x3_u8:pva_fixture_bilateral_filter_3x3_u8"
  "histogram_equalization_u8:pva_fixture_histogram_equalization_u8"
)

printf 'fixture,native,frontend,raise,linalg_generic,affine_for,scf_for,notes\n' \
  >"$OUT_DIR/summary.csv"
for item in "${fixtures[@]}"; do
  name="${item%%:*}"
  function="${item#*:}"
  frontend=pass
  raise=pass
  notes=''

  if ! cgeist "$INPUT_DIR/$name.c" --function="$function" \
      --resource-dir=/usr/lib/clang/14 --raise-scf-to-affine -S \
      -o "$OUT_DIR/$name.frontend.mlir" \
      2>"$OUT_DIR/$name.frontend.err"; then
    frontend=fail
    raise=blocked
    notes=frontend_failure
    : >"$OUT_DIR/$name.raised.mlir"
  elif ! polygeist-opt --select-func="func-name=$function" \
      --remove-iter-args --affine-parallelize \
      --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
      "$OUT_DIR/$name.frontend.mlir" -o "$OUT_DIR/$name.raised.mlir" \
      2>"$OUT_DIR/$name.raise.err"; then
    raise=fail
    notes=raising_failure
  fi

  linalg=$(grep -c 'linalg.generic' "$OUT_DIR/$name.raised.mlir" || true)
  affine=$(grep -c 'affine.for' "$OUT_DIR/$name.raised.mlir" || true)
  scf=$(grep -c 'scf.for' "$OUT_DIR/$name.raised.mlir" || true)
  printf '%s,pass,%s,%s,%s,%s,%s,%s\n' \
    "$name" "$frontend" "$raise" "$linalg" "$affine" "$scf" "$notes" \
    >>"$OUT_DIR/summary.csv"
done

cat "$OUT_DIR/summary.csv"
