#!/bin/bash
# Bake karpathy/llm.c per-function MLIR files in the naming convention the
# IR viewer expects:
#   /tmp/llmc_mlir/<tag>.mlir          (post-cgeist affine MLIR)
#   /tmp/llmc_mlir/<tag>_linalg.mlir   (after raise + lower-submap)
#   /tmp/llmc_mlir/<tag>_debuf.mlir    (default v2 debufferize)
#   /tmp/llmc_mlir/<tag>_debuf_mr.mlir (multi-root debufferize)
#
# Target the leaf forward/backward kernels in train_gpt2.c — the building
# blocks of GPT-2 inference + training. Skip the tiled matmul_forward in
# favour of matmul_forward_naive (the 4-loop reference).
set +e
_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"
SRC=$REPO_ROOT/third_party/llm.c/train_gpt2.c
OUT=/tmp/llmc_mlir
mkdir -p $OUT

# Format: <tag>  <fn>
KERNELS=(
  "encoder-fwd                encoder_forward"
  "encoder-bwd                encoder_backward"
  "layernorm-fwd              layernorm_forward"
  "layernorm-bwd              layernorm_backward"
  "matmul-fwd-naive           matmul_forward_naive"
  "matmul-bwd                 matmul_backward"
  "attention-fwd              attention_forward"
  "attention-bwd              attention_backward"
  "gelu-fwd                   gelu_forward"
  "gelu-bwd                   gelu_backward"
  "residual-fwd               residual_forward"
  "residual-bwd               residual_backward"
  "softmax-fwd                softmax_forward"
  "crossentropy-fwd           crossentropy_forward"
  "crossentropy-softmax-bwd   crossentropy_softmax_backward"
)

for entry in "${KERNELS[@]}"; do
  read tag fn <<<"$entry"

  echo "[$tag] cgeist..."
  timeout 60 cgeist "$SRC" --function=$fn --resource-dir=/usr/lib/clang/14 \
    --raise-scf-to-affine -fPIC -S \
    -o $OUT/${tag}.mlir 2>$OUT/${tag}.cgeist.err
  if [ ! -s $OUT/${tag}.mlir ]; then
    echo "  cgeist FAILED"; rm -f $OUT/${tag}.mlir; continue
  fi

  # NOTE: skip --select-func — cgeist's --function=$fn already isolated the
  # kernel, and --select-func strips extern declarations like @tanhf / @logf
  # / @expf that the math-heavy kernels call into.
  echo "[$tag] raise..."
  timeout 60 polygeist-opt \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    $OUT/${tag}.mlir -o $OUT/${tag}_linalg.mlir 2>$OUT/${tag}.raise.err
  [ ! -s $OUT/${tag}_linalg.mlir ] && { echo "  raise FAILED"; rm -f $OUT/${tag}_linalg.mlir; continue; }

  echo "[$tag] debuf v2..."
  timeout 60 polygeist-opt --linalg-debufferize \
    $OUT/${tag}_linalg.mlir -o $OUT/${tag}_debuf.mlir 2>$OUT/${tag}.debuf.err
  [ ! -s $OUT/${tag}_debuf.mlir ] && { echo "  v2 debuf FAILED"; rm -f $OUT/${tag}_debuf.mlir; }

  echo "[$tag] debuf multi-root..."
  timeout 60 polygeist-opt --linalg-debufferize=use-multi-root=true \
    $OUT/${tag}_linalg.mlir -o $OUT/${tag}_debuf_mr.mlir 2>$OUT/${tag}.debuf_mr.err
  if [ ! -s $OUT/${tag}_debuf_mr.mlir ]; then
    echo "// Multi-root --linalg-debufferize FAILED. See ${tag}.debuf_mr.err." > $OUT/${tag}_debuf_mr.mlir
  fi
done

echo "Done. Output in $OUT/"
ls $OUT/*.mlir | wc -l
