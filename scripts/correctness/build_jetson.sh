#!/bin/bash
# build_jetson.sh — compile a kernel-matched MLIR program against the real
# cuBLAS runtime for execution on a Jetson (or any x86 + NVIDIA GPU box).
#
# Prerequisites on the target machine:
#   * CUDA toolkit installed at /usr/local/cuda (or set CUDA= below)
#   * cuBLAS headers and libs (ship with the CUDA toolkit)
#   * mlir-opt / mlir-translate / clang from this Polygeist build available
#     (run scripts/build_polygeist.sh first; this typically means you ran
#     this *on* the Jetson, not cross-compiled — though cross-compile from
#     an x86 host is possible if you have NVIDIA's aarch64 cross toolkit
#     and rebuild Polygeist for aarch64. Easier path: build on-Jetson.)
#
# Usage:
#   ./build_jetson.sh <matched.mlir> <out_exe>
#
# Where <matched.mlir> is the output of `polygeist-opt --lower-kernel-launch
# -to-cublas` on a matched-MLIR module. The script handles the rest of the
# lowering, linking, and binary emission.
#
# To time + run:
#   ./<out_exe>
# Or with nsys profile:
#   nsys profile -o trace ./<out_exe>

set -euo pipefail
source /home/arjaiswal/Polygeist/envsetup.sh

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <matched.mlir> <out_exe>"
  exit 1
fi

INPUT=$1
OUT_EXE=$2
OUT_DIR=$(dirname "$OUT_EXE")
mkdir -p "$OUT_DIR"

CUDA=${CUDA:-/usr/local/cuda}
MLIR_OPT=/home/arjaiswal/Polygeist/llvm-project/build/bin/mlir-opt
MLIR_TRANSLATE=/home/arjaiswal/Polygeist/llvm-project/build/bin/mlir-translate
CLANG=/home/arjaiswal/Polygeist/llvm-project/build/bin/clang
RT=/home/arjaiswal/Polygeist/runtime

if [ ! -d "$CUDA" ]; then
  echo "ERROR: CUDA toolkit not found at $CUDA (set the CUDA env var)"
  exit 1
fi

WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

echo "  [1/5] lower-kernel-launch-to-cublas (already done? assume input is post-pass)"
cp "$INPUT" $WORK/abi.mlir

echo "  [2/5] one-shot-bufferize + lower to LLVM dialect"
sed -i 's|bufferization\.to_tensor \(%[^ ]*\) :|bufferization.to_tensor \1 restrict :|g' \
  $WORK/abi.mlir
$MLIR_OPT --one-shot-bufferize=bufferize-function-boundaries \
  --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
  --convert-arith-to-llvm --finalize-memref-to-llvm \
  --convert-func-to-llvm --reconcile-unrealized-casts \
  $WORK/abi.mlir -o $WORK/llvm.mlir

echo "  [3/5] translate to LLVM IR"
$MLIR_TRANSLATE --mlir-to-llvmir $WORK/llvm.mlir -o $WORK/kernel.ll

echo "  [4/5] compile CUDA runtime shim + kernel"
# The CUDA shim includes <cublas_v2.h> and <cuda_runtime.h>, so we need the
# CUDA include path. We compile it as C (not CUDA C++) — the headers are
# C-compatible.
$CLANG -O3 -I$CUDA/include -c $RT/polygeist_cublas_rt_cuda.c -o $WORK/rt.o
$CLANG -O3 -c $WORK/kernel.ll -o $WORK/kernel.o

echo "  [5/5] link against cuBLAS + CUDA runtime"
# Link order matters: kernel.o references runtime symbols (forward), runtime
# references cublas/cudart symbols (forward).
$CLANG $WORK/kernel.o $WORK/rt.o \
       -L$CUDA/lib64 -lcublas -lcudart \
       -lm -lpthread -ldl \
       -o "$OUT_EXE"

echo "Done. Run with:  $OUT_EXE"
echo "Profile with:    nsys profile -o ${OUT_EXE}.qdrep $OUT_EXE"
