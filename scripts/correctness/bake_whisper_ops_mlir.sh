#!/bin/bash
# Bake standalone Whisper/ggml-style operation fixtures into per-function MLIR.
#
# Outputs:
#   /tmp/whisper_ops_mlir/<tag>.mlir
#   /tmp/whisper_ops_mlir/<tag>_linalg.mlir
#   /tmp/whisper_ops_mlir/<tag>_debuf.mlir
#   /tmp/whisper_ops_mlir/<tag>_debuf_mr.mlir
#   /tmp/whisper_ops_mlir/summary.txt
set +e

_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"

SRC=$REPO_ROOT/third_party/cnn-extracted/whisper_ops.c
OUT=${POLYGEIST_WHISPER_OPS_OUT:-/tmp/whisper_ops_mlir}
CGEIST_BIN=${CGEIST_BIN:-$REPO_ROOT/build/bin/cgeist}
POLYGEIST_OPT_BIN=${POLYGEIST_OPT_BIN:-$REPO_ROOT/build/bin/polygeist-opt}

if [ -n "${POLYGEIST_CLANG_RESOURCE_DIR:-}" ]; then
  RESOURCE_DIR=$POLYGEIST_CLANG_RESOURCE_DIR
elif [ -d "$REPO_ROOT/llvm-project/build/lib/clang/18" ]; then
  RESOURCE_DIR=$REPO_ROOT/llvm-project/build/lib/clang/18
else
  RESOURCE_DIR=/usr/lib/clang/14
fi

mkdir -p "$OUT"
rm -f "$OUT"/*

# Format: <tag>  <fn>
KERNELS=(
  "whisper_vec_dot       kernel_whisper_vec_dot"
  "whisper_vec_softmax   kernel_whisper_vec_softmax"
  "whisper_softmax_full  kernel_whisper_softmax_full"
  "whisper_rms_norm      kernel_whisper_rms_norm"
  "whisper_gelu          kernel_whisper_gelu"
  "whisper_conv1d        kernel_whisper_conv1d"
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
    printf "%-24s %-18s %7s %7s %7s %7s %7s %s\n" \
      "$tag" "cgeist-fail" "-" "-" "-" "-" "-" "$OUT/${tag}.cgeist.err"
    return
  fi
  if [ ! -s "$OUT/${tag}_linalg.mlir" ]; then
    printf "%-24s %-18s %7s %7s %7s %7s %7s %s\n" \
      "$tag" "raise-fail" "-" "-" "-" "-" "-" "$OUT/${tag}.raise.err"
    return
  fi

  artifact=$(pick_artifact "$tag")
  lg=$(count_pattern "linalg\\.generic" "$artifact")
  tensor=$(count_pattern "tensor<" "$artifact")
  memref=$(count_pattern "memref<" "$artifact")
  loops=$(count_pattern "affine\\.for|scf\\.for" "$artifact")
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

  printf "%-24s %-18s %7s %7s %7s %7s %7s %s\n" \
    "$tag" "$status" "$lg" "$tensor" "$memref" "$loops" "$ifs" "$artifact"
}

SUMMARY=$OUT/summary.txt
{
  printf "%-24s %-18s %7s %7s %7s %7s %7s %s\n" \
    "op" "status" "linalg" "tensor" "memref" "loops" "ifs" "artifact"
} > "$SUMMARY"

for entry in "${KERNELS[@]}"; do
  read -r tag fn <<<"$entry"

  echo "[$tag] cgeist..."
  timeout 60 "$CGEIST_BIN" "$SRC" --function="$fn" \
    --resource-dir="$RESOURCE_DIR" --raise-scf-to-affine -fPIC -std=gnu11 -S \
    -o "$OUT/${tag}.mlir" 2>"$OUT/${tag}.cgeist.err"
  if [ ! -s "$OUT/${tag}.mlir" ]; then
    echo "  cgeist FAILED"
    rm -f "$OUT/${tag}.mlir"
    summarize_one "$tag" >> "$SUMMARY"
    continue
  fi

  echo "[$tag] raise..."
  timeout 60 "$POLYGEIST_OPT_BIN" \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    "$OUT/${tag}.mlir" -o "$OUT/${tag}_linalg.mlir" \
    2>"$OUT/${tag}.raise.err"
  if [ ! -s "$OUT/${tag}_linalg.mlir" ]; then
    echo "  raise FAILED"
    rm -f "$OUT/${tag}_linalg.mlir"
    summarize_one "$tag" >> "$SUMMARY"
    continue
  fi

  echo "[$tag] debuf v2..."
  timeout 60 "$POLYGEIST_OPT_BIN" --linalg-debufferize \
    "$OUT/${tag}_linalg.mlir" -o "$OUT/${tag}_debuf.mlir" \
    2>"$OUT/${tag}.debuf.err"
  if [ ! -s "$OUT/${tag}_debuf.mlir" ]; then
    echo "  v2 debuf FAILED"
    rm -f "$OUT/${tag}_debuf.mlir"
  fi

  echo "[$tag] debuf multi-root..."
  timeout 60 "$POLYGEIST_OPT_BIN" --linalg-debufferize=use-multi-root=true \
    "$OUT/${tag}_linalg.mlir" -o "$OUT/${tag}_debuf_mr.mlir" \
    2>"$OUT/${tag}.debuf_mr.err"
  if [ ! -s "$OUT/${tag}_debuf_mr.mlir" ]; then
    echo "  multi-root debuf FAILED"
    rm -f "$OUT/${tag}_debuf_mr.mlir"
  fi

  summarize_one "$tag" >> "$SUMMARY"
done

echo "Done. Output in $OUT"
cat "$SUMMARY"
