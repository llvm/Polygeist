#!/bin/bash
# Sweep MachSuite kernels through the Polygeist raise pipeline.
#
# For each kernel, run:
#   1. cgeist <kernel.c> --function=<fn>  →  affine MLIR
#   2. polygeist-opt --select-func=<fn> --remove-iter-args --affine-parallelize
#                    --raise-affine-to-linalg-pipeline --lower-polygeist-submap
#                    [--linalg-debufferize]
# and report: # linalg.generic, # affine.for, # scf.for after each stage.
#
# This is a coverage/diagnostic sweep — not a correctness test.
_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"
ROOT=$REPO_ROOT/third_party/MachSuite
COMMON=$ROOT/common
OUT=/tmp/machsuite_sweep
mkdir -p $OUT

# Format: <kernel-tag> <subdir> <fn>
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

# Header
printf '%-15s %5s %5s %5s   %5s %5s %5s   %5s %5s %5s   %s\n' \
  kernel CG_LG CG_AF CG_SF RS_LG RS_AF RS_SF DB_LG DB_AF DB_SF status
echo "-----------------------------------------------------------------------------------"

for entry in "${KERNELS[@]}"; do
  read tag subdir fn <<<"$entry"
  D=$ROOT/$subdir
  # Find the kernel .c (not local_support.c or generate.c)
  src=$(ls $D/*.c 2>/dev/null | grep -vE 'local_support|generate' | head -1)
  if [ -z "$src" ]; then
    printf '%-15s skipped (no source)\n' "$tag"
    continue
  fi

  # Step 1: cgeist
  cgeist "$src" --function=$fn --resource-dir=/usr/lib/clang/14 \
    -I$COMMON -I$D --raise-scf-to-affine -fPIC -S -o $OUT/${tag}.mlir \
    2>$OUT/${tag}.cgeist.err
  if [ ! -s $OUT/${tag}.mlir ]; then
    printf '%-15s   --   --   --      --   --   --      --   --   --   CGEIST_FAIL\n' "$tag"
    continue
  fi
  CG_LG=$(grep -c "linalg.generic" $OUT/${tag}.mlir 2>/dev/null); CG_LG=${CG_LG:-0}
  CG_AF=$(grep -c "affine.for" $OUT/${tag}.mlir 2>/dev/null); CG_AF=${CG_AF:-0}
  CG_SF=$(grep -c "scf.for" $OUT/${tag}.mlir 2>/dev/null); CG_SF=${CG_SF:-0}

  # Step 2: raise to linalg
  timeout 60 polygeist-opt --select-func=func-name=$fn \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    $OUT/${tag}.mlir -o $OUT/${tag}.raised.mlir 2>$OUT/${tag}.raise.err
  raise_rc=$?
  if [ "$raise_rc" -ne 0 ] || [ ! -s $OUT/${tag}.raised.mlir ]; then
    printf '%-15s %5s %5s %5s     --   --   --      --   --   --   RAISE_FAIL\n' \
      "$tag" "$CG_LG" "$CG_AF" "$CG_SF"
    continue
  fi
  RS_LG=$(grep -c "linalg.generic" $OUT/${tag}.raised.mlir 2>/dev/null); RS_LG=${RS_LG:-0}
  RS_AF=$(grep -c "affine.for" $OUT/${tag}.raised.mlir 2>/dev/null); RS_AF=${RS_AF:-0}
  RS_SF=$(grep -c "scf.for" $OUT/${tag}.raised.mlir 2>/dev/null); RS_SF=${RS_SF:-0}

  # Step 3: debufferize (multi-root)
  timeout 60 polygeist-opt --linalg-debufferize=use-multi-root=true \
    $OUT/${tag}.raised.mlir -o $OUT/${tag}.debuf.mlir 2>$OUT/${tag}.debuf.err
  debuf_rc=$?
  if [ "$debuf_rc" -ne 0 ] || [ ! -s $OUT/${tag}.debuf.mlir ]; then
    printf '%-15s %5s %5s %5s   %5s %5s %5s      --   --   --   DEBUF_FAIL\n' \
      "$tag" "$CG_LG" "$CG_AF" "$CG_SF" "$RS_LG" "$RS_AF" "$RS_SF"
    continue
  fi
  DB_LG=$(grep -c "linalg.generic" $OUT/${tag}.debuf.mlir 2>/dev/null); DB_LG=${DB_LG:-0}
  DB_AF=$(grep -c "affine.for" $OUT/${tag}.debuf.mlir 2>/dev/null); DB_AF=${DB_AF:-0}
  DB_SF=$(grep -c "scf.for" $OUT/${tag}.debuf.mlir 2>/dev/null); DB_SF=${DB_SF:-0}

  # Status classification
  if [ "$DB_LG" -gt 0 ] && [ "$DB_AF" -eq 0 ] && [ "$DB_SF" -eq 0 ]; then
    status=FULL_LIFT
  elif [ "$DB_LG" -gt 0 ]; then
    status=PARTIAL_LIFT
  else
    status=NO_LIFT
  fi
  printf '%-15s %5s %5s %5s   %5s %5s %5s   %5s %5s %5s   %s\n' \
    "$tag" "$CG_LG" "$CG_AF" "$CG_SF" "$RS_LG" "$RS_AF" "$RS_SF" \
    "$DB_LG" "$DB_AF" "$DB_SF" "$status"
done
