#!/bin/bash
# build_polybenchgpu_conv2d_jetson.sh DATASET
# Build polybenchGpu convolution-2d for one dataset, end-to-end for Jetson.
# Matches as cudnnConvolution2D_9tap_f32 (polybenchGpu DATA_TYPE defaults to float).
set -euo pipefail
_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"

DATASET=${1:?"need dataset MINI|SMALL|STANDARD|LARGE|EXTRALARGE"}

PY=$PYTHON
SCRIPTS=$REPO_ROOT/scripts/correctness
RT=$REPO_ROOT/runtime
MLIR_OPT=$REPO_ROOT/llvm-project/build/bin/mlir-opt
MLIR_TRANSLATE=$REPO_ROOT/llvm-project/build/bin/mlir-translate
CLANG=$REPO_ROOT/llvm-project/build/bin/clang

KDIR=$REPO_ROOT/third_party/polybenchGpu/OpenMP/stencils/convolution-2d
UTIL=$REPO_ROOT/third_party/polybenchGpu/OpenMP/utilities
SRC=$KDIR/convolution-2d.c
FN=kernel_conv2d
CUDA=/usr/local/cuda-12.6/targets/sbsa-linux
CUDNN_INC=/usr/include/aarch64-linux-gnu
CUDNN_LIB=/usr/lib/aarch64-linux-gnu

OUT=/tmp/conv2d_pbgpu_jetson_build
mkdir -p $OUT

echo "[conv2d/$DATASET] (1) cgeist → affine MLIR (DATA_TYPE=float default)"
cgeist $SRC --function='*' --no-inline --resource-dir=/usr/lib/clang/14 \
   -I$UTIL -I$KDIR -D${DATASET}_DATASET -Dstatic= \
   --raise-scf-to-affine -fPIC -S -o $OUT/${DATASET}_affine.mlir 2>$OUT/${DATASET}.cgeist.err
[ -s $OUT/${DATASET}_affine.mlir ] || { echo "cgeist FAIL"; head -3 $OUT/${DATASET}.cgeist.err; exit 1; }

echo "[conv2d/$DATASET] (2) raise + lower-submap (kernel only)"
polygeist-opt --select-func="func-name=$FN" \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    $OUT/${DATASET}_affine.mlir -o $OUT/${DATASET}_linalg.mlir 2>$OUT/${DATASET}.raise.err
[ -s $OUT/${DATASET}_linalg.mlir ] || { echo "raise FAIL"; head -3 $OUT/${DATASET}.raise.err; exit 1; }

echo "[conv2d/$DATASET] (3) matcher"
$PY $SCRIPTS/kernel_match_rewrite.py $OUT/${DATASET}_linalg.mlir \
    > $OUT/${DATASET}_matched.mlir 2>$OUT/${DATASET}.match.err
N_LAUNCH=$(grep -c '@cudnnConvolution2D_9tap' $OUT/${DATASET}_matched.mlir || true)
[ "${N_LAUNCH:-0}" -ge 1 ] || { echo "matcher FAIL — no cudnnConvolution2D_9tap"; exit 1; }
echo "    $N_LAUNCH conv2d_9tap launch(es)"

# Determine launch suffix (e.g. _f32). Use it for kernel.defn name + scalar type.
SUFFIX=$(grep -oE '@cudnnConvolution2D_9tap_[a-z0-9]+' $OUT/${DATASET}_matched.mlir | head -1 | sed 's/.*_//')
[ "$SUFFIX" = "f32" ] && CTYPE=float || { echo "unsupported suffix: $SUFFIX"; exit 1; }
DEFN_NAME=cudnnConvolution2D_9tap_${SUFFIX}
SCALAR_TY=$SUFFIX
echo "    using $DEFN_NAME, scalar=$SCALAR_TY"

echo "[conv2d/$DATASET] (4) inject kernel.defn for $DEFN_NAME"
$PY -c "
import sys
ty_mem = 'memref<?x?x${SCALAR_TY}, strided<[?, 1], offset: ?>>'
ty_sca = '${SCALAR_TY}'
name = '${DEFN_NAME}'
arg_list = ', '.join([f'%a{i}: {ty_mem}' for i in range(9)] + [f'%c: {ty_mem}'] + [f'%w{i}: {ty_sca}' for i in range(9)])
done = False
with open('$OUT/${DATASET}_matched.mlir') as f:
    for line in f:
        sys.stdout.write(line)
        if not done and line.startswith('module attributes'):
            print(f'  kernel.defn @{name}({arg_list}) {{ kernel.yield }}')
            done = True
" > $OUT/${DATASET}_matched_with_defn.mlir

echo "[conv2d/$DATASET] (5) lower-kernel-launch-to-cublas"
polygeist-opt --lower-kernel-launch-to-cublas \
    $OUT/${DATASET}_matched_with_defn.mlir -o $OUT/${DATASET}_abi.mlir 2>$OUT/${DATASET}.abi.err
[ -s $OUT/${DATASET}_abi.mlir ] || { echo "ABI FAIL"; head -5 $OUT/${DATASET}.abi.err; exit 1; }

# Rename + drop internal linkage so wrapper can link
sed -i "s/@${FN}\b/@${FN}_impl/g; s/llvm.linkage = #llvm.linkage<internal>//; s/func.func private @${FN}_impl/func.func @${FN}_impl/" \
    $OUT/${DATASET}_abi.mlir

echo "[conv2d/$DATASET] (6) MLIR → LLVM dialect → LLVM IR"
# Same pipeline as conv2d_cudnn_jetson.sh (not one-shot-bufferize)
$MLIR_OPT --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
    --expand-strided-metadata \
    --convert-arith-to-llvm --finalize-memref-to-llvm \
    --convert-func-to-llvm --reconcile-unrealized-casts \
    $OUT/${DATASET}_abi.mlir -o $OUT/${DATASET}_llvm.mlir 2>$OUT/${DATASET}.mlir.err
[ -s $OUT/${DATASET}_llvm.mlir ] || { echo "MLIR lower FAIL"; head -10 $OUT/${DATASET}.mlir.err; exit 1; }

$MLIR_TRANSLATE --mlir-to-llvmir $OUT/${DATASET}_llvm.mlir -o $OUT/${DATASET}_kernel.ll
sed -i 's|target triple = "x86_64.*"|target triple = "aarch64-linux-gnu"|;
        /^target datalayout/d' $OUT/${DATASET}_kernel.ll

echo "[conv2d/$DATASET] (7) cross-compile .ll → aarch64 .o"
$CLANG --target=aarch64-linux-gnu --gcc-toolchain=/usr \
    -O3 -c $OUT/${DATASET}_kernel.ll -o $OUT/${DATASET}_kernel.o 2>&1 | tail -3

echo "[conv2d/$DATASET] (8) cross-compile harness + wrapper + rt"
HARNESS_CFLAGS=(-O3 -I"$UTIL" -I"$KDIR"
                -DPOLYBENCH_DUMP_ARRAYS -D${DATASET}_DATASET -Dstatic=
                -DPOLYBENCH_USE_C99_PROTO)
ARCH_FLAGS="-march=armv8.2-a+fp16+bf16"

aarch64-linux-gnu-gcc "${HARNESS_CFLAGS[@]}" -c "$SRC" -o $OUT/${DATASET}_full.o
aarch64-linux-gnu-objcopy --weaken-symbol=$FN $OUT/${DATASET}_full.o $OUT/${DATASET}_nokernel.o
aarch64-linux-gnu-gcc "${HARNESS_CFLAGS[@]}" -c "$UTIL/polybench.c" -o $OUT/${DATASET}_polybench.o
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS -DCTYPE=$CTYPE -c $SCRIPTS/conv2d_jetson_wrapper_dtype.c -o $OUT/${DATASET}_wrapper.o
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS -I$CUDA/include -I$CUDNN_INC -c $RT/polygeist_cublas_rt_cuda.c -o $OUT/${DATASET}_rt_cuda.o

echo "[conv2d/$DATASET] (9) link"
aarch64-linux-gnu-gcc -O2 \
    $OUT/${DATASET}_kernel.o $OUT/${DATASET}_rt_cuda.o \
    $OUT/${DATASET}_wrapper.o $OUT/${DATASET}_nokernel.o $OUT/${DATASET}_polybench.o \
    -L$CUDA/lib -L$CUDA/lib/stubs -L$CUDNN_LIB \
    -lcudnn -lcublasLt -lcublas -lcudart -lm -lpthread -ldl \
    -Wl,-rpath,/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu \
    -o $OUT/conv2d_jetson_${DATASET}

echo "OK: $OUT/conv2d_jetson_${DATASET}"
ls -l $OUT/conv2d_jetson_${DATASET}
