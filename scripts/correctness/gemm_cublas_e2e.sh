#!/bin/bash
# gemm_cublas_e2e.sh — end-to-end test of the Phase-2 cuBLAS-ABI lowering.
#
# Pipeline:
#   1. C source                        (gemm.c, MINI_DATASET)
#   2. cgeist                          → affine MLIR
#   3. polygeist-opt raise + debuf     → tensor-form linalg.generic
#   4. kernel_match_rewrite.py         → tensor-form with kernel.launch ops
#   5. polygeist-opt --lower-kernel-launch-to-cublas
#                                       → tensor-form with func.call to
#                                         polygeist_cublas_dgemm (runtime shim)
#   6. mlir-opt one-shot-bufferize + std lowerings → LLVM dialect
#   7. mlir-translate                  → LLVM IR
#   8. clang -c                        → kernel.o
#   9. link with polygeist_cublas_rt_cpu.o (CPU stub) + polybench harness
#  10. run, diff vs clang -O0 reference
#
# On a real GPU/Jetson, swap step 9 to link against polygeist_cublas_rt_cuda.o
# + -lcublas -lcudart (see build_jetson.sh).
#
# Pass = "matched kernel.launch through cuBLAS-ABI runtime shim produces the
# same numeric output as the clang reference build".

set -euo pipefail
_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"

MLIR_OPT=$REPO_ROOT/llvm-project/build/bin/mlir-opt
MLIR_TRANSLATE=$REPO_ROOT/llvm-project/build/bin/mlir-translate
CLANG=$REPO_ROOT/llvm-project/build/bin/clang
PYTHON=$PYTHON
SCRIPTS=$REPO_ROOT/scripts/correctness
RT=$REPO_ROOT/runtime

POLYBENCH_DIR=$REPO_ROOT/tools/cgeist/Test/polybench
UTIL=$POLYBENCH_DIR/utilities
GEMM_DIR=$POLYBENCH_DIR/linear-algebra/blas/gemm

OUT=/tmp/gemm_cublas_test
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
grep -c "func.func @kernel_gemm" $OUT/gemm_orig.mlir > /dev/null

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

echo "  c) kernel-match (linalg -> kernel.launch)"
$PYTHON $SCRIPTS/kernel_match_rewrite.py \
  $OUT/gemm_debuf.mlir > $OUT/gemm_matched.mlir 2>$OUT/match.err
N_LAUNCH=$(grep -c '= kernel\.launch ' $OUT/gemm_matched.mlir || echo 0)
echo "    matched ops: $N_LAUNCH kernel.launch"
if [ "$N_LAUNCH" -lt 1 ]; then
  echo "  FAIL: expected at least 1 kernel.launch"; exit 1
fi

echo "  d) inject kernel.defn declaration (verifier needs the symbol to exist)"
# The matched MLIR refers to @cublasDgemm but does not define it. Without a
# `kernel.defn`, the parser's symbol-user verifier rejects the kernel.launch
# ops. We inject a trivial defn body (just yields the C operand) — our pass
# never reads the body, only the symbol; it's deleted again post-lowering.
awk '/^module attributes/ && !done{
       print;
       print "  kernel.defn @cublasDgemm(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>, %C: tensor<?x?xf64>, %beta: f64, %alpha: f64) -> tensor<?x?xf64> {";
       print "    kernel.yield %C : tensor<?x?xf64>";
       print "  }";
       done=1;
       next
     }{print}' $OUT/gemm_matched.mlir > $OUT/gemm_matched_with_defn.mlir

echo "  e) lower-kernel-launch-to-cublas (kernel.launch -> func.call ABI)"
polygeist-opt --lower-kernel-launch-to-cublas \
  $OUT/gemm_matched_with_defn.mlir -o $OUT/gemm_abi.mlir 2>$OUT/abi.err
N_LAUNCH_AFTER=$(grep -c '= kernel\.launch ' $OUT/gemm_abi.mlir 2>/dev/null || true)
N_CALL=$(grep -cE 'call @polygeist_cublas_dgemm\(' $OUT/gemm_abi.mlir 2>/dev/null || true)
N_LAUNCH_AFTER=${N_LAUNCH_AFTER:-0}
N_CALL=${N_CALL:-0}
echo "    residual kernel.launch: $N_LAUNCH_AFTER ; func.call to shim: $N_CALL"
if [ "$N_LAUNCH_AFTER" -ne 0 ] || [ "$N_CALL" -lt 1 ]; then
  echo "  FAIL: lowering didn't replace kernel.launch with the runtime call"
  cat $OUT/abi.err
  exit 1
fi

echo "  f) lower to LLVM dialect"
# Mark to_tensor results as `restrict` so one-shot-bufferize knows it's safe
# to keep the in-place semantics (same trick gemm_kernel_e2e.sh uses).
sed -i 's|bufferization\.to_tensor \(%[^ ]*\) :|bufferization.to_tensor \1 restrict :|g' \
  $OUT/gemm_abi.mlir
$MLIR_OPT --one-shot-bufferize=bufferize-function-boundaries \
  --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
  --convert-arith-to-llvm --finalize-memref-to-llvm \
  --convert-func-to-llvm --reconcile-unrealized-casts \
  $OUT/gemm_abi.mlir -o $OUT/gemm_llvm.mlir 2>$OUT/mlir.err

echo "  g) translate to LLVM IR"
$MLIR_TRANSLATE --mlir-to-llvmir $OUT/gemm_llvm.mlir -o $OUT/gemm.ll 2>$OUT/translate.err
sed -i 's/@kernel_gemm\b/@kernel_gemm_impl/g' $OUT/gemm.ll

echo "  h) compile runtime shim + harness pieces"
$CLANG -O2 -c $RT/polygeist_cublas_rt_cpu.c -o $OUT/rt.o
$CLANG -c $CFLAGS $DYN_FLAGS $GEMM_DIR/gemm.c -o $OUT/gemm_full.o
objcopy --weaken-symbol=kernel_gemm $OUT/gemm_full.o $OUT/gemm_nokernel.o
$CLANG -c $CFLAGS $UTIL/polybench.c -o $OUT/polybench.o
$CLANG -c $SCRIPTS/gemm_wrapper.c -o $OUT/wrapper.o
$CLANG -c $OUT/gemm.ll -o $OUT/kernel.o

echo "  i) link (CPU-stub runtime, no CUDA)"
$CLANG $OUT/gemm_nokernel.o $OUT/wrapper.o $OUT/kernel.o $OUT/polybench.o \
       $OUT/rt.o -lm -o $OUT/test_exe

echo "=== 3. Run test and diff ==="
$OUT/test_exe 2> $OUT/test.out
wc -l $OUT/test.out

if diff -q $OUT/ref.out $OUT/test.out >/dev/null; then
  echo "PASS: cuBLAS-ABI lowering e2e matches clang reference"
else
  echo "FAIL: outputs differ"
  diff $OUT/ref.out $OUT/test.out | head -10
  exit 1
fi
