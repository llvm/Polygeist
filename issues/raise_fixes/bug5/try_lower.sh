#!/bin/bash
# usage: try_lower.sh KDIR(e2e_v2 kernel dir) IN.mlir TAG [pre-passes...]
# lowers IN.mlir (memref or tensor form) with e2e_v2's LOWER_TO_LLVM and links
# with the kernel's harness objects; prints the comparison vs ref.out
KD=$1; IN=$2; TAG=$3; shift 3
R=/home/arjaiswal/Polygeist; L=$R/llvm-project/build
FN=$(grep -o "func.func @kernel_[a-z0-9_]*" $IN | head -1 | sed 's/func.func @//')
set -e
if [ $# -gt 0 ]; then $R/build-v2/bin/polygeist-opt "$@" $IN -o $TAG.pre.mlir; else cp $IN $TAG.pre.mlir; fi
sed -i 's|bufferization\.to_tensor \(%[^ ]*\) :|bufferization.to_tensor \1 restrict :|g' $TAG.pre.mlir
$L/bin/mlir-opt --convert-math-to-llvm --empty-tensor-to-alloc-tensor --lower-affine --one-shot-bufferize=bufferize-function-boundaries --convert-linalg-to-loops --convert-scf-to-cf --expand-strided-metadata --lower-affine --convert-arith-to-llvm --convert-index-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts $TAG.pre.mlir -o $TAG.llvm.mlir
$L/bin/mlir-translate --mlir-to-llvmir $TAG.llvm.mlir -o $TAG.ll
sed -i "s/@$FN\b/@${FN}_impl/g" $TAG.ll
$L/bin/clang -O1 -g $KD/harness.o $KD/wrapper.o $KD/polybench.o $TAG.ll -lm -L$L/lib -Wl,-rpath,$L/lib -lmlir_c_runner_utils -o $TAG.exe 2>/dev/null
set +e
./$TAG.exe > /dev/null 2> $TAG.out; rc=$?
if cmp -s $TAG.out $KD/ref.out; then echo "$TAG: rc=$rc IDENTICAL"; else echo "$TAG: rc=$rc DIFF ($(wc -c < $TAG.out) vs $(wc -c < $KD/ref.out) bytes)"; fi
