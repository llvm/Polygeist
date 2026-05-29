#!/bin/bash
set -e
_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"
MLIR_OPT=$REPO_ROOT/llvm-project/build/bin/mlir-opt
MLIR_TRANSLATE=$REPO_ROOT/llvm-project/build/bin/mlir-translate
CLANG=$REPO_ROOT/llvm-project/build/bin/clang

POLYBENCH_DIR=$REPO_ROOT/tools/cgeist/Test/polybench
UTIL=$POLYBENCH_DIR/utilities
GEMM_DIR=$POLYBENCH_DIR/linear-algebra/blas/gemm

OUT=/tmp/gemm_debuf_test
mkdir -p $OUT

DATASET=-DMINI_DATASET   # 20x25x30 — small for fast iteration
CFLAGS="-O1 -I$UTIL -I$GEMM_DIR -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_DUMP_ARRAYS $DATASET"
# Use C99 prototypes + suppress static-size hints so cgeist produces fully-
# dynamic memrefs that round-trip cleanly through --linalg-debufferize.
DYN_FLAGS="-Dstatic= -DPOLYBENCH_USE_C99_PROTO"

echo "=== 1. Reference: clang -O0 directly ==="
$CLANG $CFLAGS $DYN_FLAGS \
  $GEMM_DIR/gemm.c $UTIL/polybench.c -lm -o $OUT/ref_exe
$OUT/ref_exe 2> $OUT/ref.out
wc -l $OUT/ref.out

echo "=== 2. Test pipeline ==="
echo "  a) cgeist gemm.c -> MLIR"
cgeist $GEMM_DIR/gemm.c --function=kernel_gemm --resource-dir=/usr/lib/clang/14 \
  $CFLAGS $DYN_FLAGS --raise-scf-to-affine -S -o $OUT/gemm_orig.mlir 2>/dev/null
grep -c "func.func @kernel_gemm" $OUT/gemm_orig.mlir

echo "  b) raise + lower-polygeist-submap"
polygeist-opt --select-func=func-name=kernel_gemm \
  --remove-iter-args --affine-parallelize \
  --raise-affine-to-linalg-pipeline \
  --lower-polygeist-submap \
  --linalg-debufferize \
  $OUT/gemm_orig.mlir -o $OUT/gemm_std.mlir 2>$OUT/raise.err
# Check no polygeist ops remain
if grep -qE "polygeist\.(submap|submapInverse)" $OUT/gemm_std.mlir; then
  echo "  FAIL: polygeist ops remain"; exit 1
fi
echo "    raise+lower OK"

echo "  c) lower to LLVM dialect"
# bufferization.to_tensor needs `restrict` for one-shot-bufferize to accept
# it. The LinalgDebufferize pass doesn't emit this attr, so patch via sed.
sed -i 's|bufferization\.to_tensor \(%[^ ]*\) :|bufferization.to_tensor \1 restrict :|g' \
  $OUT/gemm_std.mlir
$MLIR_OPT --one-shot-bufferize=bufferize-function-boundaries \
  --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
  --convert-arith-to-llvm --finalize-memref-to-llvm \
  --convert-func-to-llvm --reconcile-unrealized-casts \
  $OUT/gemm_std.mlir -o $OUT/gemm_llvm.mlir 2>$OUT/mlir.err

echo "  d) translate to LLVM IR"
$MLIR_TRANSLATE --mlir-to-llvmir $OUT/gemm_llvm.mlir -o $OUT/gemm.ll 2>$OUT/translate.err
# Rename the lowered function so our wrapper can name it
sed -i 's/@kernel_gemm\b/@kernel_gemm_impl/g' $OUT/gemm.ll

echo "  e) compile gemm.c with kernel_gemm SUPPRESSED (we'll provide our own)"
# Trick: use the preprocessor to rename gemm.c's kernel_gemm into a static
# function (then it's defined-but-private, and our extern kernel_gemm wins).
# But macro replaces both definition and call. So instead, compile gemm.c
# to gemm.o with the kernel intact, then objcopy --strip-symbol the
# kernel_gemm symbol. After strip the call from main becomes an undef ref,
# which our wrapper.o satisfies.
$CLANG -c $CFLAGS $DYN_FLAGS $GEMM_DIR/gemm.c -o $OUT/gemm_full.o
# Rename the definition's symbol to a stub; main's relocation still points
# to kernel_gemm, which our wrapper.o will satisfy.
objcopy --redefine-sym kernel_gemm=__unused_kernel_gemm \
  $OUT/gemm_full.o $OUT/gemm_nokernel.o
# But the call from main also got renamed — undo that by re-redefining
# the call site... actually --redefine-sym renames ALL occurrences. So main
# also calls __unused_kernel_gemm now. Wrong. We need to instead rename
# only the DEFINITION, not the references. objcopy doesn't distinguish.
# Workaround: use a linker script or weakening.
objcopy --weaken-symbol=kernel_gemm $OUT/gemm_full.o $OUT/gemm_nokernel.o

echo "  f) compile polybench.c"
$CLANG -c $CFLAGS $UTIL/polybench.c -o $OUT/polybench.o

echo "  g) compile wrapper + lowered kernel"
$CLANG -c /tmp/gemm_wrapper.c -o $OUT/wrapper.o
$CLANG -c $OUT/gemm.ll -o $OUT/kernel.o

echo "  h) link"
$CLANG $OUT/gemm_nokernel.o $OUT/wrapper.o $OUT/kernel.o $OUT/polybench.o -lm -o $OUT/test_exe

echo "=== 3. Run test and diff ==="
$OUT/test_exe 2> $OUT/test.out
wc -l $OUT/test.out

echo "=== diff ==="
if diff -q $OUT/ref.out $OUT/test.out; then
  echo "PASS: outputs match"
else
  echo "FAIL: outputs differ"
  diff $OUT/ref.out $OUT/test.out | head -10
  exit 1
fi
