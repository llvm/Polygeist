#!/bin/bash
# First-pass cgeist/linalg probes for selected C ECP proxy apps.
set +e

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OUT=${POLYGEIST_PROXY_APP_OUT:-/tmp/proxy_app_raise_mlir}
CGEIST_BIN=${CGEIST_BIN:-$REPO_ROOT/build/bin/cgeist}
POLYGEIST_OPT_BIN=${POLYGEIST_OPT_BIN:-$REPO_ROOT/build/bin/polygeist-opt}
PROBE_INCLUDE=$REPO_ROOT/issues/proxy_app_raise_probes/include

if [ -n "${POLYGEIST_CLANG_RESOURCE_DIR:-}" ]; then
  RESOURCE_DIR=$POLYGEIST_CLANG_RESOURCE_DIR
elif [ -d "$REPO_ROOT/llvm-project/build/lib/clang/18" ]; then
  RESOURCE_DIR=$REPO_ROOT/llvm-project/build/lib/clang/18
else
  RESOURCE_DIR=/usr/lib/clang/14
fi

mkdir -p "$OUT"
rm -f "$OUT"/*

count_pattern() {
  local pattern=$1
  local file=$2
  if [ ! -s "$file" ]; then
    echo 0
    return
  fi
  grep -Ec "$pattern" "$file" 2>/dev/null
}

pick_artifact() {
  local tag=$1
  if [ -s "$OUT/${tag}_debuf_mr.mlir" ] &&
     grep -q "linalg.generic" "$OUT/${tag}_debuf_mr.mlir"; then
    echo "$OUT/${tag}_debuf_mr.mlir"
  elif [ -s "$OUT/${tag}_debuf.mlir" ] &&
       grep -q "linalg.generic" "$OUT/${tag}_debuf.mlir"; then
    echo "$OUT/${tag}_debuf.mlir"
  elif [ -s "$OUT/${tag}_linalg.mlir" ]; then
    echo "$OUT/${tag}_linalg.mlir"
  else
    echo "$OUT/${tag}.mlir"
  fi
}

summarize_one() {
  local tag=$1
  local status artifact lg tensor memref loops ifs

  if [ ! -s "$OUT/${tag}.mlir" ]; then
    printf "%-32s %-18s %7s %7s %7s %7s %7s %s\n" \
      "$tag" "cgeist-fail" "-" "-" "-" "-" "-" "$OUT/${tag}.cgeist.err"
    return
  fi
  if [ ! -s "$OUT/${tag}_linalg.mlir" ]; then
    printf "%-32s %-18s %7s %7s %7s %7s %7s %s\n" \
      "$tag" "raise-fail" "-" "-" "-" "-" "-" "$OUT/${tag}.raise.err"
    return
  fi

  artifact=$(pick_artifact "$tag")
  lg=$(count_pattern "linalg\\.generic" "$artifact")
  tensor=$(count_pattern "tensor<" "$artifact")
  memref=$(count_pattern "memref<" "$artifact")
  loops=$(count_pattern "affine\\.for|scf\\.for|affine\\.parallel|scf\\.parallel" "$artifact")
  ifs=$(count_pattern "affine\\.if|scf\\.if" "$artifact")

  if [ "$lg" -gt 0 ] && [ "$tensor" -gt 0 ]; then
    status="tensor-linalg"
  elif [ "$lg" -gt 0 ]; then
    status="memref-linalg"
  else
    status="no-linalg"
  fi
  if [ "$loops" -gt 0 ]; then
    status="${status}+loops"
  fi
  if [ "$ifs" -gt 0 ]; then
    status="${status}+if"
  fi

  printf "%-32s %-18s %7s %7s %7s %7s %7s %s\n" \
    "$tag" "$status" "$lg" "$tensor" "$memref" "$loops" "$ifs" "$artifact"
}

run_probe() {
  local tag=$1
  local src=$2
  local cgeist_fn=$3
  local select_fn=$4
  shift 4

  echo "[$tag] cgeist $cgeist_fn"
  timeout 90 "$CGEIST_BIN" "$REPO_ROOT/$src" --function="$cgeist_fn" \
    --resource-dir="$RESOURCE_DIR" --raise-scf-to-affine -fPIC -std=gnu11 -S \
    "$@" -o "$OUT/${tag}.mlir" 2>"$OUT/${tag}.cgeist.err"
  if [ ! -s "$OUT/${tag}.mlir" ]; then
    echo "  cgeist FAILED"
    rm -f "$OUT/${tag}.mlir"
    summarize_one "$tag" >> "$SUMMARY"
    return
  fi

  local select_args=()
  if [ "$select_fn" != "-" ]; then
    select_args=(--select-func=func-name="$select_fn")
  fi

  echo "[$tag] raise"
  timeout 90 "$POLYGEIST_OPT_BIN" \
    "${select_args[@]}" \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    "$OUT/${tag}.mlir" -o "$OUT/${tag}_linalg.mlir" \
    2>"$OUT/${tag}.raise.err"
  if [ ! -s "$OUT/${tag}_linalg.mlir" ]; then
    echo "  raise FAILED"
    rm -f "$OUT/${tag}_linalg.mlir"
    summarize_one "$tag" >> "$SUMMARY"
    return
  fi

  echo "[$tag] debuf v2"
  timeout 90 "$POLYGEIST_OPT_BIN" --linalg-debufferize \
    "$OUT/${tag}_linalg.mlir" -o "$OUT/${tag}_debuf.mlir" \
    2>"$OUT/${tag}.debuf.err"
  if [ ! -s "$OUT/${tag}_debuf.mlir" ]; then
    rm -f "$OUT/${tag}_debuf.mlir"
  fi

  echo "[$tag] debuf multi-root"
  timeout 90 "$POLYGEIST_OPT_BIN" --linalg-debufferize=use-multi-root=true \
    "$OUT/${tag}_linalg.mlir" -o "$OUT/${tag}_debuf_mr.mlir" \
    2>"$OUT/${tag}.debuf_mr.err"
  if [ ! -s "$OUT/${tag}_debuf_mr.mlir" ]; then
    rm -f "$OUT/${tag}_debuf_mr.mlir"
  fi

  summarize_one "$tag" >> "$SUMMARY"
}

SUMMARY=$OUT/summary.txt
printf "%-32s %-18s %7s %7s %7s %7s %7s %s\n" \
  "probe" "status" "linalg" "tensor" "memref" "loops" "ifs" "artifact" > "$SUMMARY"

run_probe "swfft_distribution_2_to_3" \
  "third_party/SWFFT/distribution.c" "distribution_2_to_3" "-" \
  -I"$PROBE_INCLUDE" -I"$REPO_ROOT/third_party/SWFFT" \
  -DNDEBUG -include "$PROBE_INCLUDE/swfft_probe_disable_debug.h"

run_probe "miniamr_stencil_calc" \
  "third_party/miniAMR/ref/stencil.c" "stencil_calc" "-" \
  -I"$PROBE_INCLUDE" -I"$REPO_ROOT/third_party/miniAMR/ref"

run_probe "hypar_linear_adr_advection" \
  "third_party/hypar/src/PhysicalModels/LinearADR/LinearADRAdvection.c" \
  "LinearADRAdvection" "LinearADRAdvection" \
  -I"$PROBE_INCLUDE" -I"$REPO_ROOT/third_party/hypar/include"

run_probe "hpgmg_7pt_apply_op" \
  "third_party/hpgmg/finite-volume/source/operators.7pt.c" "*" "-" \
  -I"$PROBE_INCLUDE" -I"$REPO_ROOT/third_party/hpgmg/finite-volume/source" \
  -DUSE_JACOBI

run_probe "exasp2_sp2_loop" \
  "third_party/ExaSP2/src/sp2Basic.c" "sp2Loop" "-" \
  -I"$PROBE_INCLUDE" -I"$REPO_ROOT/third_party/ExaSP2/src" \
  -DSP2_BASIC -DNTIMING -DNCOUNTING

echo "Done. Output in $OUT"
cat "$SUMMARY"
