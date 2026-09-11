#!/bin/bash
# bake_darknet_mlir.sh — try lifting every .c file in third_party/darknet/src/
# through cgeist + raise + match, and report which ones produce useful
# linalg.generic / kernel.launch ops.
#
# Goal: empirically see how many of darknet's 46 source files contain
# patterns our matcher can recognize. Predicted outcome: ~3 useful
# (gemm.c, im2col.c, maybe blas.c). The rest is framework code with no
# compute loops the raise pass can hoist.
set +e
_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"

ROOT=$REPO_ROOT/third_party/darknet
OUT=/tmp/darknet_mlir
PY=$PYTHON
SCRIPTS=$REPO_ROOT/scripts/correctness
mkdir -p $OUT

# Track results
TOTAL=0
CGEIST_OK=0
RAISE_OK=0
MATCH_OK=0
HAS_LINALG=0

# Header
printf "%-30s %-7s %-7s %-6s %-6s %s\n" "file" "cgeist" "raise" "lg" "match" "callees"
printf "%-30s %-7s %-7s %-6s %-6s %s\n" "----" "------" "-----" "--" "-----" "-------"

for src in $ROOT/src/*.c; do
  base=$(basename "$src" .c)
  TOTAL=$((TOTAL+1))

  # Skip CUDA-only files (.c that uses CUDA API directly)
  if grep -q "cudaMalloc\|cublas\|cudnn" "$src" 2>/dev/null && [ "$base" = "cuda" ]; then
    printf "%-30s %-7s %-7s %-6s %-6s %s\n" "$base" "SKIP" "-" "-" "-" "(cuda.c)"
    continue
  fi

  # 1. cgeist — emit affine MLIR for every function. Keep inlining enabled so
  # same-translation-unit helper calls are exposed before the raise pipeline;
  # --raise-scf-to-affine gives us affine.for nests where possible.
  affine=$OUT/${base}.affine.mlir
  timeout 60 cgeist "$src" --function='*' \
      --resource-dir=/usr/lib/clang/14 \
      -I$ROOT/include -I$ROOT/src \
      --raise-scf-to-affine -fPIC -S \
      -o $affine 2>$OUT/${base}.cgeist.err
  if [ ! -s "$affine" ]; then
    printf "%-30s %-7s %-7s %-6s %-6s %s\n" "$base" "FAIL" "-" "-" "-" "$(head -1 $OUT/${base}.cgeist.err 2>/dev/null | head -c 60)"
    continue
  fi
  CGEIST_OK=$((CGEIST_OK+1))

  # 2. raise — try to emit linalg.generic. We run without --select-func
  # because we don't know which function holds the compute kernel; the
  # raise pipeline is applied module-wide.
  linalg=$OUT/${base}.linalg.mlir
  timeout 60 polygeist-opt \
      --remove-iter-args --affine-parallelize \
      --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
      --linalg-debufferize \
      $affine -o $linalg 2>$OUT/${base}.raise.err
  if [ ! -s "$linalg" ]; then
    printf "%-30s %-7s %-7s %-6s %-6s %s\n" "$base" "OK" "FAIL" "-" "-" "$(head -1 $OUT/${base}.raise.err 2>/dev/null | head -c 60)"
    continue
  fi
  RAISE_OK=$((RAISE_OK+1))

  # Count linalg.generic ops
  lg=$(grep -c "linalg.generic" $linalg 2>/dev/null)
  lg=${lg:-0}
  if [ "$lg" -gt 0 ]; then HAS_LINALG=$((HAS_LINALG+1)); fi

  # 3. matcher
  matched=$OUT/${base}.matched.mlir
  timeout 60 $PY $SCRIPTS/kernel_match_rewrite.py $linalg > $matched 2>$OUT/${base}.match.err
  klc=$(grep -c "kernel.launch" $matched 2>/dev/null)
  klc=${klc:-0}
  if [ "$klc" -gt 0 ]; then MATCH_OK=$((MATCH_OK+1)); fi

  callees=$(grep -oE "kernel.launch @[A-Za-z0-9_]+" $matched 2>/dev/null | sort -u | sed 's|kernel.launch @||' | tr '\n' ',' | sed 's/,$//')

  printf "%-30s %-7s %-7s %-6d %-6d %s\n" "$base" "OK" "OK" "$lg" "$klc" "${callees:--}"
done

echo ""
echo "═══ Summary ═══"
echo "Total .c files:                   $TOTAL"
echo "cgeist succeeded:                 $CGEIST_OK"
echo "raise succeeded:                  $RAISE_OK"
echo "files with ≥1 linalg.generic:     $HAS_LINALG"
echo "files with ≥1 kernel.launch:      $MATCH_OK"
