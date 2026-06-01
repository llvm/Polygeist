#!/bin/bash
# Bake image/PDE-style 2D stencil fixtures and run the kernel matcher.
#
# Outputs:
#   /tmp/stencil_conv2d_mlir/<tag>.mlir
#   /tmp/stencil_conv2d_mlir/<tag>_linalg.mlir
#   /tmp/stencil_conv2d_mlir/<tag>_debuf.mlir
#   /tmp/stencil_conv2d_mlir/<tag>_debuf_mr.mlir
#   /tmp/stencil_conv2d_mlir/<tag>_matched.mlir
#   /tmp/stencil_conv2d_mlir/summary.txt
set +e

_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"

SRC=$REPO_ROOT/third_party/cnn-extracted/stencil_conv2d_3x3.c
OUT=${POLYGEIST_STENCIL_CONV2D_OUT:-/tmp/stencil_conv2d_mlir}
mkdir -p "$OUT"
rm -f "$OUT"/*

if ! "$PYTHON" -c "import egglog" >/dev/null 2>&1; then
  if /usr/bin/python3 -c "import egglog" >/dev/null 2>&1; then
    PYTHON=/usr/bin/python3
  fi
fi

# Format: <tag>  <function>
KERNELS=(
  "box3x3          kernel_stencil_box3x3"
  "gaussian3x3     kernel_stencil_gaussian3x3"
  "sobel_x3x3      kernel_stencil_sobel_x3x3"
  "sobel_y3x3      kernel_stencil_sobel_y3x3"
  "laplacian4_3x3  kernel_stencil_laplacian4_3x3"
  "laplacian8_3x3  kernel_stencil_laplacian8_3x3"
  "sharpen3x3      kernel_stencil_sharpen3x3"
  "emboss3x3       kernel_stencil_emboss3x3"
  "box5x5          kernel_stencil_box5x5"
  "gaussian5x5     kernel_stencil_gaussian5x5"
  "sobel_x5x5      kernel_stencil_sobel_x5x5"
  "sobel_y5x5      kernel_stencil_sobel_y5x5"
  "laplacian5x5    kernel_stencil_laplacian5x5"
  "sharpen5x5      kernel_stencil_sharpen5x5"
  "emboss5x5       kernel_stencil_emboss5x5"
)

count_pattern() {
  local pattern=$1
  local file=$2
  if [ ! -s "$file" ]; then
    echo 0
    return
  fi
  grep -Ec "$pattern" "$file" 2>/dev/null
}

match_symbol() {
  local file=$1
  if [ ! -s "$file" ]; then
    echo "-"
    return
  fi
  "$PYTHON" "$SCRIPTS/kernel_match_rewrite.py" "$file" --dry-run \
    2>&1 |
    tee "$file.match.err" |
    awk '/match[[:space:]]+body#/ {print $3}' |
    paste -sd "," -
}

summary=$OUT/summary.txt
printf "%-16s %-12s %7s %7s %7s %-36s %s\n" \
  "kernel" "status" "linalg" "loops" "launch" "matched-symbol" "artifact" > "$summary"

for entry in "${KERNELS[@]}"; do
  read -r tag fn <<<"$entry"
  echo "[$tag] cgeist..."
  timeout 60 cgeist "$SRC" --function="$fn" --resource-dir=/usr/lib/clang/14 \
    --raise-scf-to-affine -fPIC -S \
    -o "$OUT/${tag}.mlir" 2>"$OUT/${tag}.cgeist.err"
  if [ ! -s "$OUT/${tag}.mlir" ]; then
    printf "%-16s %-12s %7s %7s %7s %-36s %s\n" \
      "$tag" "cgeist-fail" "-" "-" "-" "-" "$OUT/${tag}.cgeist.err" >> "$summary"
    echo "  cgeist FAILED"
    continue
  fi

  echo "[$tag] raise..."
  timeout 60 polygeist-opt --select-func=func-name="$fn" \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    "$OUT/${tag}.mlir" -o "$OUT/${tag}_linalg.mlir" \
    2>"$OUT/${tag}.raise.err"
  if [ ! -s "$OUT/${tag}_linalg.mlir" ]; then
    printf "%-16s %-12s %7s %7s %7s %-36s %s\n" \
      "$tag" "raise-fail" "-" "-" "-" "-" "$OUT/${tag}.raise.err" >> "$summary"
    echo "  raise FAILED"
    continue
  fi

  echo "[$tag] debuf v2..."
  timeout 60 polygeist-opt --linalg-debufferize \
    "$OUT/${tag}_linalg.mlir" -o "$OUT/${tag}_debuf.mlir" \
    2>"$OUT/${tag}.debuf.err"
  [ ! -s "$OUT/${tag}_debuf.mlir" ] && rm -f "$OUT/${tag}_debuf.mlir"

  echo "[$tag] debuf multi-root..."
  timeout 60 polygeist-opt --linalg-debufferize=use-multi-root=true \
    "$OUT/${tag}_linalg.mlir" -o "$OUT/${tag}_debuf_mr.mlir" \
    2>"$OUT/${tag}.debuf_mr.err"
  [ ! -s "$OUT/${tag}_debuf_mr.mlir" ] && rm -f "$OUT/${tag}_debuf_mr.mlir"

  "$PYTHON" "$SCRIPTS/kernel_match_rewrite.py" "$OUT/${tag}_linalg.mlir" \
    > "$OUT/${tag}_matched.mlir" 2>"$OUT/${tag}.match.err"

  lg=$(count_pattern "linalg\\.generic" "$OUT/${tag}_linalg.mlir")
  loops=$(count_pattern "affine\\.for|scf\\.for" "$OUT/${tag}_linalg.mlir")
  launches=$(count_pattern "kernel\\.launch" "$OUT/${tag}_matched.mlir")
  sym=$(match_symbol "$OUT/${tag}_linalg.mlir")
  [ -z "$sym" ] && sym="-"
  status="matched"
  [ "$launches" -eq 0 ] && status="no-match"

  printf "%-16s %-12s %7s %7s %7s %-36s %s\n" \
    "$tag" "$status" "$lg" "$loops" "$launches" "$sym" \
    "$OUT/${tag}_linalg.mlir" >> "$summary"
done

echo "Done. Output in $OUT"
cat "$summary"
