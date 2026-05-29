#!/bin/bash
# Bake MachSuite per-kernel MLIR files in the naming convention the IR
# viewer expects:
#   /tmp/machsuite_mlir/<tag>.mlir          (post-cgeist affine MLIR)
#   /tmp/machsuite_mlir/<tag>_linalg.mlir   (after raise + lower-submap)
#   /tmp/machsuite_mlir/<tag>_debuf.mlir    (default v2 debufferize)
#   /tmp/machsuite_mlir/<tag>_debuf_mr.mlir (multi-root debufferize)
#
# Kernels that don't produce a given stage are skipped silently — viewer's
# `if file.exists():` branches handle missing files gracefully.
set +e
_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"
ROOT=$REPO_ROOT/third_party/MachSuite
COMMON=$ROOT/common
OUT=/tmp/machsuite_mlir
mkdir -p $OUT

# Format: <tag> <subdir> <fn>  (same map as machsuite_sweep.sh)
KERNELS=(
  "aes              aes/aes              aes256_encrypt_ecb"
  "backprop         backprop/backprop    backprop"
  "bfs-bulk         bfs/bulk             bfs"
  "bfs-queue        bfs/queue            bfs"
  "fft-strided      fft/strided          fft"
  "fft-transpose    fft/transpose        fft1D_512"
  "gemm-ncubed      gemm/ncubed          gemm"
  "gemm-blocked     gemm/blocked         bbgemm"
  "kmp              kmp/kmp              kmp"
  "md-grid          md/grid              md"
  "md-knn           md/knn               md_kernel"
  "nw               nw/nw                needwun"
  "sort-merge       sort/merge           ms_mergesort"
  "sort-radix       sort/radix           ss_sort"
  "spmv-crs         spmv/crs             spmv"
  "spmv-ellpack     spmv/ellpack         ellpack"
  "stencil2d        stencil/stencil2d    stencil"
  "stencil3d        stencil/stencil3d    stencil3d"
  "viterbi          viterbi/viterbi      viterbi"
)

for entry in "${KERNELS[@]}"; do
  read tag subdir fn <<<"$entry"
  D=$ROOT/$subdir
  src=$(ls $D/*.c 2>/dev/null | grep -vE 'local_support|generate' | head -1)
  [ -z "$src" ] && continue

  echo "[$tag] cgeist..."
  cgeist "$src" --function=$fn --resource-dir=/usr/lib/clang/14 \
    -I$COMMON -I$D --raise-scf-to-affine -fPIC -S -o $OUT/${tag}.mlir \
    2>$OUT/${tag}.cgeist.err
  [ ! -s $OUT/${tag}.mlir ] && { echo "  cgeist FAILED"; rm -f $OUT/${tag}.mlir; continue; }

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
ls $OUT/ | head -20
