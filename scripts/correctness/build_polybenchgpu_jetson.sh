#!/bin/bash
# build_polybenchgpu_jetson.sh KERNEL DATASET
# Build a single polybenchGpu kernel for one dataset size, end-to-end.
# Produces /tmp/<KERNEL>_pbgpu_jetson_build/<KERNEL>_jetson_<DATASET>
set -euo pipefail
_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"

KERNEL=${1:?"need kernel name e.g. syrk"}
DATASET=${2:?"need dataset e.g. MINI|LARGE|EXTRALARGE"}

PY=$PYTHON
SCRIPTS=$REPO_ROOT/scripts/correctness
MLIR_OPT=$REPO_ROOT/llvm-project/build/bin/mlir-opt
MLIR_TRANSLATE=$REPO_ROOT/llvm-project/build/bin/mlir-translate

ROOT=$REPO_ROOT/third_party/polybenchGpu/OpenMP
UTIL=$ROOT/utilities
# Find the kernel subdir
case "$KERNEL" in
  syrk|gemm|gemver|gesummv|2mm|3mm|atax|bicg|mvt|symm|syr2k|trmm|trisolv) KDIR=$ROOT/linear-algebra/kernels/$KERNEL ;;
  convolution-2d|convolution-3d|fdtd-2d|fdtd-apml|jacobi-1d-imper|jacobi-2d-imper|seidel-2d|adi) KDIR=$ROOT/stencils/$KERNEL ;;
  correlation|covariance) KDIR=$ROOT/datamining/$KERNEL ;;
  *) echo "ERROR: unknown kernel $KERNEL" >&2; exit 1 ;;
esac

SRC=$(ls $KDIR/*.c 2>/dev/null | head -1)
[ -z "$SRC" ] && { echo "ERROR: no .c in $KDIR" >&2; exit 1; }

FN="kernel_${KERNEL//-/_}"

OUT=/tmp/${KERNEL}_pbgpu_jetson_build
mkdir -p $OUT

HARNESS_CFLAGS=(-O3 -I"$UTIL" -I"$KDIR"
                -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_DUMP_ARRAYS
                -D${DATASET}_DATASET -Dstatic= -DPOLYBENCH_USE_C99_PROTO)
# cgeist flags — note polybenchGpu's old polybench.h breaks if we pass
# POLYBENCH_USE_C99_PROTO to cgeist, so we DON'T (the static dim baked in
# will match the dataset because we set -D${DATASET}_DATASET).
CGEIST_FLAGS=(-I"$UTIL" -I"$KDIR" -DDATA_TYPE_IS_DOUBLE
              -D${DATASET}_DATASET -Dstatic=
              --resource-dir=/usr/lib/clang/14
              --raise-scf-to-affine -fPIC -S)

echo "[$KERNEL/$DATASET] (1) cgeist → affine MLIR"
cgeist "$SRC" --function='*' --no-inline "${CGEIST_FLAGS[@]}" \
  -o $OUT/${DATASET}_affine.mlir 2>$OUT/${DATASET}.cgeist.err
[ -s $OUT/${DATASET}_affine.mlir ] || { echo "cgeist FAIL"; head -3 $OUT/${DATASET}.cgeist.err; exit 1; }

echo "[$KERNEL/$DATASET] (2) raise + debuf"
polygeist-opt --select-func="func-name=$FN" \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    --linalg-debufferize \
    $OUT/${DATASET}_affine.mlir -o $OUT/${DATASET}_debuf.mlir 2>$OUT/${DATASET}.raise.err
[ -s $OUT/${DATASET}_debuf.mlir ] || { echo "raise FAIL"; head -3 $OUT/${DATASET}.raise.err; exit 1; }

echo "[$KERNEL/$DATASET] (3) matcher: linalg → kernel.launch"
$PY $SCRIPTS/kernel_match_rewrite.py $OUT/${DATASET}_debuf.mlir \
    > $OUT/${DATASET}_matched.mlir 2>$OUT/${DATASET}.match.err
N_LAUNCH=$(grep -c "kernel.launch" $OUT/${DATASET}_matched.mlir || true)
echo "    matched $N_LAUNCH kernel.launch ops"
[ "${N_LAUNCH:-0}" -ge 1 ] || { echo "matcher FAIL"; exit 1; }

echo "[$KERNEL/$DATASET] (4) inject kernel.defn @cublasDgemm + lower-kernel-launch-to-cublas"
# Determine the static second dim from the matched MLIR
SECOND_DIM=$(grep -oE "tensor<\?x[0-9]+xf64>" $OUT/${DATASET}_matched.mlir | head -1 | sed -E 's/tensor<\?x([0-9]+)xf64>/\1/')
[ -z "$SECOND_DIM" ] && { echo "Couldn't determine static second dim"; exit 1; }
echo "    static second dim: $SECOND_DIM"
TY="tensor<?x${SECOND_DIM}xf64>"

$PY -c "
import sys
ty = '$TY'
done = False
with open('$OUT/${DATASET}_matched.mlir') as f:
    for line in f:
        sys.stdout.write(line)
        if not done and line.startswith('module attributes'):
            print(f'  kernel.defn @cublasDgemm(%A: {ty}, %B: {ty}, %C: {ty}, %beta: f64, %alpha: f64) -> {ty} {{')
            print(f'    kernel.yield %C : {ty}')
            print('  }')
            done = True
" > $OUT/${DATASET}_matched_with_defn.mlir
sed -i 's/!any/f64/g' $OUT/${DATASET}_matched_with_defn.mlir

polygeist-opt --lower-kernel-launch-to-cublas \
    $OUT/${DATASET}_matched_with_defn.mlir -o $OUT/${DATASET}_abi.mlir 2>$OUT/${DATASET}.abi.err
[ -s $OUT/${DATASET}_abi.mlir ] || { echo "ABI lower FAIL"; head -3 $OUT/${DATASET}.abi.err; exit 1; }

# Rename kernel function + drop internal linkage
sed -i "s/@${FN}\b/@${FN}_impl/g; s/llvm.linkage = #llvm.linkage<internal>//; s/func.func private @${FN}_impl/func.func @${FN}_impl/" \
    $OUT/${DATASET}_abi.mlir

echo "[$KERNEL/$DATASET] (5) cross-compile harness"
aarch64-linux-gnu-gcc "${HARNESS_CFLAGS[@]}" -c "$SRC" -o $OUT/${DATASET}_full.o
aarch64-linux-gnu-objcopy --weaken-symbol=$FN $OUT/${DATASET}_full.o $OUT/${DATASET}_nokernel.o
aarch64-linux-gnu-gcc "${HARNESS_CFLAGS[@]}" -c "$UTIL/polybench.c" -o $OUT/${DATASET}_polybench.o

echo "[$KERNEL/$DATASET] (6) build_jetson.sh → aarch64 binary"
bash $SCRIPTS/build_jetson.sh \
    $OUT/${DATASET}_abi.mlir \
    $OUT/${KERNEL}_jetson_${DATASET} \
    $SCRIPTS/${KERNEL}_jetson_wrapper.c \
    $OUT/${DATASET}_nokernel.o \
    $OUT/${DATASET}_polybench.o 2>&1 | tail -3

echo "OK: $OUT/${KERNEL}_jetson_${DATASET}"
