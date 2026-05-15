#!/bin/bash
# End-to-end correctness test: C source -> ... -> kernel.launch (matched) ->
# lower-kernel-launch (restored linalg) -> LLVM dialect -> binary -> execute.
#
# Compares numeric output against a pure clang reference. Pass = round-trip
# through the kernel-match form preserves the gemm computation.
#
# Phase 1: roundtrip lowering — we restore the matcher's pre-match linalg
# verbatim from comment markers. This validates that match-then-lower doesn't
# corrupt the SSA chain or surrounding IR, and that the e2e plumbing works.
# It does NOT validate the matcher's library LABEL ("@cublasDgemm"); that's
# Phase 2 (canonical templates).
set -euo pipefail
source /home/arjaiswal/Polygeist/envsetup.sh

MLIR_OPT=/home/arjaiswal/Polygeist/llvm-project/build/bin/mlir-opt
MLIR_TRANSLATE=/home/arjaiswal/Polygeist/llvm-project/build/bin/mlir-translate
CLANG=/home/arjaiswal/Polygeist/llvm-project/build/bin/clang
PYTHON=/home/arjaiswal/slacker/.venv/bin/python3
SCRIPTS=/home/arjaiswal/Polygeist/scripts/correctness

POLYBENCH_DIR=/home/arjaiswal/Polygeist/tools/cgeist/Test/polybench
UTIL=$POLYBENCH_DIR/utilities
GEMM_DIR=$POLYBENCH_DIR/linear-algebra/blas/gemm

OUT=/tmp/gemm_kernel_test
mkdir -p $OUT

DATASET=-DMINI_DATASET
CFLAGS="-O1 -I$UTIL -I$GEMM_DIR -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_DUMP_ARRAYS $DATASET"
DYN_FLAGS="-Dstatic= -DPOLYBENCH_USE_C99_PROTO"

echo "=== 1. Reference: clang -O0 directly ==="
$CLANG $CFLAGS $DYN_FLAGS \
  $GEMM_DIR/gemm.c $UTIL/polybench.c -lm -o $OUT/ref_exe
$OUT/ref_exe 2> $OUT/ref.out
wc -l $OUT/ref.out

echo "=== 2. Test pipeline ==="
echo "  a) cgeist gemm.c -> affine MLIR"
cgeist $GEMM_DIR/gemm.c --function=kernel_gemm --resource-dir=/usr/lib/clang/14 \
  $CFLAGS $DYN_FLAGS --raise-scf-to-affine -S -o $OUT/gemm_orig.mlir 2>/dev/null
grep -c "func.func @kernel_gemm" $OUT/gemm_orig.mlir

echo "  b) raise + lower-submap + debufferize"
polygeist-opt --select-func=func-name=kernel_gemm \
  --remove-iter-args --affine-parallelize \
  --raise-affine-to-linalg-pipeline \
  --lower-polygeist-submap \
  --linalg-debufferize \
  $OUT/gemm_orig.mlir -o $OUT/gemm_debuf.mlir 2>$OUT/raise.err
if grep -qE "polygeist\.(submap|submapInverse)" $OUT/gemm_debuf.mlir; then
  echo "  FAIL: polygeist ops remain after lower-submap"; exit 1
fi

echo "  c) kernel-match (linalg -> kernel.launch, with roundtrip markers)"
$PYTHON $SCRIPTS/kernel_match_rewrite.py --with-roundtrip-markers \
  $OUT/gemm_debuf.mlir > $OUT/gemm_matched.mlir 2>$OUT/match.err
N_LAUNCH=$(grep -c '= kernel\.launch ' $OUT/gemm_matched.mlir || echo 0)
N_MARK=$(grep -c '// POLYGEIST-MATCH-BEGIN-' $OUT/gemm_matched.mlir || echo 0)
echo "    matched ops: $N_LAUNCH kernel.launch, $N_MARK markers"
if [ "$N_LAUNCH" -lt 1 ] || [ "$N_MARK" -ne "$N_LAUNCH" ]; then
  echo "  FAIL: expected at least 1 kernel.launch and matching markers"; exit 1
fi

echo "  d) lower-kernel-launch (kernel.launch -> restored linalg)"
$PYTHON $SCRIPTS/kernel_launch_lower.py $OUT/gemm_matched.mlir \
  -o $OUT/gemm_restored.mlir 2>$OUT/lower.err
# Sanity: restored output must be bit-exact to the pre-match debufferized IR.
if ! diff -q $OUT/gemm_debuf.mlir $OUT/gemm_restored.mlir >/dev/null; then
  echo "  FAIL: restored MLIR is not bit-exact to pre-match"
  diff -u $OUT/gemm_debuf.mlir $OUT/gemm_restored.mlir | head -30
  exit 1
fi
echo "    restoration bit-exact OK"

echo "  e) lower to LLVM dialect"
sed -i 's|bufferization\.to_tensor \(%[^ ]*\) :|bufferization.to_tensor \1 restrict :|g' \
  $OUT/gemm_restored.mlir
$MLIR_OPT --one-shot-bufferize=bufferize-function-boundaries \
  --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
  --convert-arith-to-llvm --finalize-memref-to-llvm \
  --convert-func-to-llvm --reconcile-unrealized-casts \
  $OUT/gemm_restored.mlir -o $OUT/gemm_llvm.mlir 2>$OUT/mlir.err

echo "  f) translate to LLVM IR"
$MLIR_TRANSLATE --mlir-to-llvmir $OUT/gemm_llvm.mlir -o $OUT/gemm.ll 2>$OUT/translate.err
sed -i 's/@kernel_gemm\b/@kernel_gemm_impl/g' $OUT/gemm.ll

echo "  g) compile gemm.c with kernel_gemm weakened"
$CLANG -c $CFLAGS $DYN_FLAGS $GEMM_DIR/gemm.c -o $OUT/gemm_full.o
objcopy --weaken-symbol=kernel_gemm $OUT/gemm_full.o $OUT/gemm_nokernel.o

echo "  h) compile polybench + wrapper + lowered kernel"
$CLANG -c $CFLAGS $UTIL/polybench.c -o $OUT/polybench.o
$CLANG -c $SCRIPTS/gemm_wrapper.c -o $OUT/wrapper.o
$CLANG -c $OUT/gemm.ll -o $OUT/kernel.o

echo "  i) link"
$CLANG $OUT/gemm_nokernel.o $OUT/wrapper.o $OUT/kernel.o $OUT/polybench.o \
  -lm -o $OUT/test_exe

echo "=== 3. Run test and diff ==="
$OUT/test_exe 2> $OUT/test.out
wc -l $OUT/test.out

if diff -q $OUT/ref.out $OUT/test.out >/dev/null; then
  echo "PASS: kernel.launch roundtrip e2e outputs match clang reference"
else
  echo "FAIL: outputs differ"
  diff $OUT/ref.out $OUT/test.out | head -10
  exit 1
fi
