#!/bin/bash
# Bake llama2.c per-function MLIR files in the naming convention the IR
# viewer expects:
#   /tmp/llama2c_mlir/<tag>.mlir          (post-cgeist affine MLIR)
#   /tmp/llama2c_mlir/<tag>_linalg.mlir   (after raise + lower-submap)
#   /tmp/llama2c_mlir/<tag>_debuf.mlir    (default v2 debufferize)
#   /tmp/llama2c_mlir/<tag>_debuf_mr.mlir (multi-root debufferize)
#
# Target the hot numeric functions in run.c. Other functions (tokenizer,
# I/O, sampling) are not interesting for raising.
set +e
source /home/arjaiswal/Polygeist/envsetup.sh
SRC=/home/arjaiswal/Polygeist/third_party/llama2.c/run.c
OUT=/tmp/llama2c_mlir
mkdir -p $OUT

# Format: <tag>  <fn>
KERNELS=(
  "rmsnorm   rmsnorm"
  "softmax   softmax"
  "matmul    matmul"
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

  echo "[$tag] raise..."
  timeout 60 polygeist-opt --select-func=func-name=$fn \
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
ls $OUT/ | head -30
