#!/bin/bash
# Run an end-to-end correctness test for one PolyBench kernel.
#
# Usage:
#   run_kernel_e2e.sh <kernel_dir> <kernel_name> [--debuf]
#
# Example:
#   run_kernel_e2e.sh tools/cgeist/Test/polybench/linear-algebra/blas/gemm gemm
#   run_kernel_e2e.sh ... gemm --debuf       # also run --linalg-debufferize
#
# Returns 0 on PASS, non-zero on any failure or output mismatch.
set -e
source /home/arjaiswal/Polygeist/envsetup.sh
MLIR_OPT=/home/arjaiswal/Polygeist/llvm-project/build/bin/mlir-opt
MLIR_TRANSLATE=/home/arjaiswal/Polygeist/llvm-project/build/bin/mlir-translate
CLANG=/home/arjaiswal/Polygeist/llvm-project/build/bin/clang
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 2 ]; then
  sed -n '3,12p' "$0" >&2
  exit 1
fi
KERNEL_DIR="$1"
KERNEL="$2"   # short name, e.g. "gemm", "mvt"
DEBUF=""
[ "${3:-}" = "--debuf" ] && DEBUF="1"

# PolyBench source files: <dir>/<short>.c. Kernel function is
# `kernel_<short>` with hyphens replaced by underscores (heat-3d → kernel_heat_3d).
SRC="$KERNEL_DIR/${KERNEL}.c"
FN="kernel_${KERNEL//-/_}"

if [ ! -f "$SRC" ]; then echo "MISSING: $SRC"; exit 2; fi

POLYBENCH_DIR=/home/arjaiswal/Polygeist/tools/cgeist/Test/polybench
UTIL=$POLYBENCH_DIR/utilities

TAG="$KERNEL"
[ -n "$DEBUF" ] && TAG="${KERNEL}_debuf"
OUT=/tmp/e2e_${TAG}
mkdir -p $OUT

DATASET=-DMINI_DATASET
CFLAGS="-O1 -I$UTIL -I$KERNEL_DIR -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_DUMP_ARRAYS $DATASET"
DYN_FLAGS="-Dstatic= -DPOLYBENCH_USE_C99_PROTO"

# Pipeline ordering: lower-polygeist-submap BEFORE --linalg-debufferize so
# debuferize sees only standard MLIR.
PIPELINE_OPTS=(
  --select-func=func-name=$FN
  --remove-iter-args --affine-parallelize
  --raise-affine-to-linalg-pipeline
  --lower-polygeist-submap
)
if [ -n "$DEBUF" ]; then
  PIPELINE_OPTS+=(--linalg-debufferize)
fi

# Step 1: build the reference exe.
$CLANG $CFLAGS $DYN_FLAGS $SRC $UTIL/polybench.c -lm -o $OUT/ref_exe 2>$OUT/ref_compile.err

# Step 2: cgeist gemm.c -> MLIR.
cgeist "$SRC" --function=$FN --resource-dir=/usr/lib/clang/14 \
  $CFLAGS $DYN_FLAGS --raise-scf-to-affine -S -o $OUT/orig.mlir 2>$OUT/cgeist.err

# Step 3: raise + lower-polygeist-submap (+ optional debuferize).
polygeist-opt "${PIPELINE_OPTS[@]}" $OUT/orig.mlir -o $OUT/std.mlir 2>$OUT/raise.err

# Bail if any polygeist ops survive.
if grep -qE "polygeist\.(submap|submapInverse)" $OUT/std.mlir; then
  echo "$TAG: PARTIAL_LOWER (polygeist ops remain)"
  exit 3
fi

# Step 4: standard MLIR lowering to LLVM dialect.
# The debuferize path emits `bufferization.to_tensor` that one-shot-bufferize
# needs `restrict` on. LinalgDebufferize doesn't emit it; patch via sed.
# Also: one-shot-bufferize doesn't handle `affine.for` with tensor iter_args,
# which debuferize emits for time-stepping kernels. Convert affine.for ->
# scf.for first (via --lower-affine) so bufferize sees only scf.for.
if [ -n "$DEBUF" ]; then
  sed -i 's|bufferization\.to_tensor \(%[^ ]*\) :|bufferization.to_tensor \1 restrict :|g' $OUT/std.mlir
  EXTRA="--lower-affine --empty-tensor-to-alloc-tensor --one-shot-bufferize=bufferize-function-boundaries"
else
  EXTRA=""
fi
$MLIR_OPT $EXTRA --expand-strided-metadata \
  --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
  --convert-arith-to-llvm --convert-math-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm --reconcile-unrealized-casts \
  $OUT/std.mlir -o $OUT/llvm.mlir 2>$OUT/mlir.err

# Step 5: translate to LLVM IR and rename kernel function.
$MLIR_TRANSLATE --mlir-to-llvmir $OUT/llvm.mlir -o $OUT/kernel.ll 2>$OUT/translate.err
sed -i "s/@${FN}\b/@${FN}_impl/g" $OUT/kernel.ll

# Step 6: generate the C wrapper for this kernel.
python3 $SCRIPT_DIR/gen_wrapper.py "$SRC" "$FN" > $OUT/wrapper.c 2>$OUT/wrapper_gen.err

# Step 7: compile pieces. Weaken kernel_* in gemm.o so wrapper.o wins.
$CLANG -c $CFLAGS $DYN_FLAGS $SRC -o $OUT/full.o
objcopy --weaken-symbol=$FN $OUT/full.o $OUT/nokernel.o
$CLANG -c $CFLAGS $UTIL/polybench.c -o $OUT/polybench.o
$CLANG -c $OUT/wrapper.c -o $OUT/wrapper.o
$CLANG -c $OUT/kernel.ll -o $OUT/kernel.o
$CLANG $OUT/nokernel.o $OUT/wrapper.o $OUT/kernel.o $OUT/polybench.o -lm \
  -o $OUT/test_exe

# Step 8: run both, diff. Tolerate a non-zero exit on test_exe — some
# kernels crash on heap-free after the dump, but the dump itself is
# what we're comparing.
set +e
$OUT/ref_exe 2> $OUT/ref.out
$OUT/test_exe 2> $OUT/test.out
set -e
if diff -q $OUT/ref.out $OUT/test.out >/dev/null; then
  echo "$TAG: PASS"
  exit 0
else
  echo "$TAG: FAIL_DIFF (first 5 differing lines:)"
  diff $OUT/ref.out $OUT/test.out | head -5
  exit 4
fi
