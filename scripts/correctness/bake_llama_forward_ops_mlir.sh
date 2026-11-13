#!/bin/bash
# Bake standalone Llama-forward operation fixtures into per-function MLIR.
#
# Outputs:
#   /tmp/llama_forward_ops_mlir/<tag>.mlir
#   /tmp/llama_forward_ops_mlir/<tag>_linalg.mlir
#   /tmp/llama_forward_ops_mlir/<tag>_debuf.mlir
#   /tmp/llama_forward_ops_mlir/<tag>_debuf_mr.mlir
#   /tmp/llama_forward_ops_mlir/summary.txt
#
# The summary is a quick triage of whether each operation reached linalg and
# whether any debufferized artifact contains tensor linalg.
set +e

_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"

SRC=$REPO_ROOT/third_party/cnn-extracted/llama_forward_ops.c
EXTENDED_SRC=$REPO_ROOT/third_party/cnn-extracted/llama2_extended_forward_bench.c
OUT=${POLYGEIST_LLAMA_OPS_OUT:-/tmp/llama_forward_ops_mlir}
EXTENDED_TIMEOUT=${POLYGEIST_LLAMA_EXTENDED_TIMEOUT:-180}
mkdir -p "$OUT"
rm -f "$OUT"/*

# Format: <tag>  <fn>
KERNELS=(
  "token_embedding       kernel_llama_token_embedding"
  "attention_rmsnorm     kernel_llama_attention_rmsnorm"
  "qkv_projection        kernel_llama_qkv_projection"
  "rope_interleaved      kernel_llama_rope"
  "rope_split            kernel_llama_rope_split"
  "kv_cache_rw           kernel_llama_kv_cache_rw"
  "attention_scores      kernel_llama_attention_scores"
  "attention_mask_if     kernel_llama_attention_mask"
  "attention_mask_select kernel_llama_attention_mask_select"
  "attention_softmax     kernel_llama_attention_softmax"
  "attention_output      kernel_llama_attention_output"
  "output_projection     kernel_llama_output_projection"
  "residual_add          kernel_llama_residual_add"
  "ffn_rmsnorm           kernel_llama_ffn_rmsnorm"
  "gate_up_projection    kernel_llama_gate_up_projection"
  "swiglu                kernel_llama_swiglu"
  "down_projection       kernel_llama_down_projection"
  "final_rmsnorm         kernel_llama_final_rmsnorm"
  "lm_head_projection    kernel_llama_lm_head_projection"
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
    printf "%-22s %-17s %7s %7s %7s %7s %7s %s\n" \
      "$tag" "cgeist-fail" "-" "-" "-" "-" "-" "$OUT/${tag}.cgeist.err"
    return
  fi
  if [ ! -s "$OUT/${tag}_linalg.mlir" ]; then
    printf "%-22s %-17s %7s %7s %7s %7s %7s %s\n" \
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

  printf "%-22s %-17s %7s %7s %7s %7s %7s %s\n" \
    "$tag" "$status" "$lg" "$tensor" "$memref" "$loops" "$ifs" "$artifact"
}

SUMMARY=$OUT/summary.txt
{
  printf "%-22s %-17s %7s %7s %7s %7s %7s %s\n" \
    "op" "status" "linalg" "tensor" "memref" "loops" "ifs" "artifact"
} > "$SUMMARY"

for entry in "${KERNELS[@]}"; do
  read -r tag fn <<<"$entry"

  echo "[$tag] cgeist..."
  timeout 60 cgeist "$SRC" --function="$fn" --resource-dir=/usr/lib/clang/14 \
    --raise-scf-to-affine -fPIC -S \
    -o "$OUT/${tag}.mlir" 2>"$OUT/${tag}.cgeist.err"
  if [ ! -s "$OUT/${tag}.mlir" ]; then
    echo "  cgeist FAILED"
    rm -f "$OUT/${tag}.mlir"
    summarize_one "$tag" >> "$SUMMARY"
    continue
  fi

  echo "[$tag] raise..."
  timeout 60 polygeist-opt --select-func=func-name="$fn" \
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
  timeout 60 polygeist-opt --linalg-debufferize \
    "$OUT/${tag}_linalg.mlir" -o "$OUT/${tag}_debuf.mlir" \
    2>"$OUT/${tag}.debuf.err"
  if [ ! -s "$OUT/${tag}_debuf.mlir" ]; then
    echo "  v2 debuf FAILED"
    rm -f "$OUT/${tag}_debuf.mlir"
  fi

  echo "[$tag] debuf multi-root..."
  timeout 60 polygeist-opt --linalg-debufferize=use-multi-root=true \
    "$OUT/${tag}_linalg.mlir" -o "$OUT/${tag}_debuf_mr.mlir" \
    2>"$OUT/${tag}.debuf_mr.err"
  if [ ! -s "$OUT/${tag}_debuf_mr.mlir" ]; then
    echo "  multi-root debuf FAILED"
    rm -f "$OUT/${tag}_debuf_mr.mlir"
  fi

  summarize_one "$tag" >> "$SUMMARY"
done

tag=extended_forward
fn=kernel_llama2_extended_forward

echo "[$tag] cgeist..."
timeout "$EXTENDED_TIMEOUT" cgeist "$EXTENDED_SRC" --function="$fn" --resource-dir=/usr/lib/clang/14 \
  --raise-scf-to-affine -fPIC -S \
  -o "$OUT/${tag}.mlir" 2>"$OUT/${tag}.cgeist.err"
if [ ! -s "$OUT/${tag}.mlir" ]; then
  echo "  cgeist FAILED"
  rm -f "$OUT/${tag}.mlir"
  summarize_one "$tag" >> "$SUMMARY"
else
  echo "[$tag] raise..."
  timeout "$EXTENDED_TIMEOUT" polygeist-opt --select-func=func-name="$fn" \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    "$OUT/${tag}.mlir" -o "$OUT/${tag}_linalg.mlir" \
    2>"$OUT/${tag}.raise.err"
  if [ ! -s "$OUT/${tag}_linalg.mlir" ]; then
    echo "  raise FAILED"
    rm -f "$OUT/${tag}_linalg.mlir"
    summarize_one "$tag" >> "$SUMMARY"
  else
    echo "[$tag] debuf v2..."
    timeout "$EXTENDED_TIMEOUT" polygeist-opt --linalg-debufferize \
      "$OUT/${tag}_linalg.mlir" -o "$OUT/${tag}_debuf.mlir" \
      2>"$OUT/${tag}.debuf.err"
    if [ ! -s "$OUT/${tag}_debuf.mlir" ]; then
      echo "  v2 debuf FAILED"
      rm -f "$OUT/${tag}_debuf.mlir"
    fi

    echo "[$tag] debuf multi-root..."
    timeout "$EXTENDED_TIMEOUT" polygeist-opt --linalg-debufferize=use-multi-root=true \
      "$OUT/${tag}_linalg.mlir" -o "$OUT/${tag}_debuf_mr.mlir" \
      2>"$OUT/${tag}.debuf_mr.err"
    if [ ! -s "$OUT/${tag}_debuf_mr.mlir" ]; then
      echo "  multi-root debuf FAILED"
      rm -f "$OUT/${tag}_debuf_mr.mlir"
    fi

    summarize_one "$tag" >> "$SUMMARY"
  fi
fi

echo "Done. Output in $OUT"
cat "$SUMMARY"
