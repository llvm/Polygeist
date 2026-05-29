#!/bin/bash
# pva_bilateral_jetson.sh — end-to-end test of the OpBilateralFilter PVA path.
# Skips the matcher (which doesn't yet emit pvaBilateralFilter_*) and hand-
# authors the kernel.launch directly, then runs the same lowering +
# cross-compile + Jetson silicon validation pipeline as the conv2d tests.
#
# Usage: ./pva_bilateral_jetson.sh <DTYPE> [SIZE]
#   <DTYPE>: i8 | i16
#   [SIZE]:  default 256
#
# Output: /tmp/pva_bilateral_<DTYPE>_<SIZE>/{bilateral_jetson, bilateral_jetson_cpustub}

set -euo pipefail
_CORRECTNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_CORRECTNESS_DIR/common_env.sh"

DTYPE=${1:?"missing DTYPE arg (i8|i16)"}
SIZE=${2:-256}
SCRIPTS=$REPO_ROOT/scripts/correctness
RT=$REPO_ROOT/runtime
OUT=/tmp/pva_bilateral_${DTYPE}_${SIZE}
mkdir -p $OUT
CUDA=/usr/local/cuda-12.6/targets/sbsa-linux

case "$DTYPE" in
  i8)  MTY=i8;  CTY=int8_t;  ;;
  i16) MTY=i16; CTY=int16_t; ;;
  *) echo "unknown dtype: $DTYPE"; exit 1;;
esac

echo "[bilateral/$DTYPE/$SIZE] (1) author kernel.launch MLIR by hand"
cat > $OUT/synth.mlir <<EOF
module {
  kernel.defn @pvaBilateralFilter_3x3_${DTYPE}(
      %a: memref<?x?x${MTY}, strided<[?, 1], offset: ?>>,
      %b: memref<?x?x${MTY}, strided<[?, 1], offset: ?>>) {
    kernel.yield
  }
  func.func @kernel_conv2d(%ni: i32, %nj: i32,
                            %A: memref<?x${SIZE}x${MTY}>,
                            %B: memref<?x${SIZE}x${MTY}>)
      attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %ni_idx = arith.index_cast %ni : i32 to index
    %nj_idx = arith.index_cast %nj : i32 to index
    %m2 = arith.subi %ni_idx, %c2 : index
    %n2 = arith.subi %nj_idx, %c2 : index
    %Av = memref.subview %A[0, 0] [%m2, %n2] [1, 1]
        : memref<?x${SIZE}x${MTY}> to memref<?x?x${MTY}, strided<[${SIZE}, 1]>>
    %Bv = memref.subview %B[1, 1] [%m2, %n2] [1, 1]
        : memref<?x${SIZE}x${MTY}> to memref<?x?x${MTY}, strided<[${SIZE}, 1], offset: $((SIZE + 1))>>
    %Ac = memref.cast %Av
        : memref<?x?x${MTY}, strided<[${SIZE}, 1]>>
       to memref<?x?x${MTY}, strided<[?, 1], offset: ?>>
    %Bc = memref.cast %Bv
        : memref<?x?x${MTY}, strided<[${SIZE}, 1], offset: $((SIZE + 1))>>
       to memref<?x?x${MTY}, strided<[?, 1], offset: ?>>
    kernel.launch @pvaBilateralFilter_3x3_${DTYPE}(%Ac, %Bc)
        : (memref<?x?x${MTY}, strided<[?, 1], offset: ?>>,
           memref<?x?x${MTY}, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}
EOF

echo "[bilateral/$DTYPE/$SIZE] (2) lower-kernel-launch-to-pva"
polygeist-opt --lower-kernel-launch-to-pva $OUT/synth.mlir -o $OUT/abi.mlir 2>$OUT/abi.err

echo "[bilateral/$DTYPE/$SIZE] (3) lower to LLVM, translate, retarget aarch64"
MLIR_OPT=$REPO_ROOT/llvm-project/build/bin/mlir-opt
MLIR_TRANSLATE=$REPO_ROOT/llvm-project/build/bin/mlir-translate
CLANG=$REPO_ROOT/llvm-project/build/bin/clang
$MLIR_OPT --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
    --expand-strided-metadata \
    --convert-arith-to-llvm --finalize-memref-to-llvm \
    --convert-func-to-llvm --reconcile-unrealized-casts \
    $OUT/abi.mlir -o $OUT/llvm.mlir 2>$OUT/mlir.err
$MLIR_TRANSLATE --mlir-to-llvmir $OUT/llvm.mlir -o $OUT/kernel.ll
sed -i 's|target triple = "x86_64.*"|target triple = "aarch64-linux-gnu"|;
        /^target datalayout/d;
        s/@kernel_conv2d\b/@kernel_conv2d_impl/g' $OUT/kernel.ll
$CLANG --target=aarch64-linux-gnu --gcc-toolchain=/usr \
    -O3 -c $OUT/kernel.ll -o $OUT/kernel.o 2>&1 | tail -1

echo "[bilateral/$DTYPE/$SIZE] (4) cross-compile harness + wrapper + runtimes"
ARCH_FLAGS="-march=armv8.2-a+fp16+bf16"
KIND_DEF="-DCTYPE_KIND_INT"
DEFS="-DNI=$SIZE -DNJ=$SIZE -DCTYPE=$CTY $KIND_DEF"
PVASOL_INC=${PVASOL_INC:-$PVASOL_ROOT/public/src/operator/include}
NVCV_INC=${NVCV_INC:-$CV_CUDA_ROOT/src/nvcv/src/include}
CUPVA_INC=${CUPVA_INC:-$CUPVA_SDK_ROOT/include}
PVA_LIB_STAGE=${PVA_LIB_STAGE:-$HOME/pva_libs}
JET_PVA_LIB=/tmp/pva_libs

aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS $DEFS -c $SCRIPTS/conv2d_main_harness_dtype.c -o $OUT/main.o
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS -DCTYPE=$CTY -c $SCRIPTS/conv2d_jetson_wrapper_dtype.c -o $OUT/wrapper.o
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS -c $RT/polygeist_cublas_rt_cpu.c -o $OUT/rt_cpu.o
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS \
    -I$CUDA/include -I$PVASOL_INC -I$NVCV_INC -I$CUPVA_INC \
    -c $RT/polygeist_pva_rt.c -o $OUT/rt_pva.o
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS -I$CUDA/include -c $RT/polygeist_cublas_rt_cuda.c -o $OUT/rt_cuda.o

echo "[bilateral/$DTYPE/$SIZE] (5) link PVA binary"
PVA_LINK="-L$PVA_LIB_STAGE -lpva_operator -lcvcuda -lnvcv_types -lcupva_host \
          -Wl,--no-as-needed \
          -L$JETSON_NVIDIA_LIBS -lnvscibuf -lnvscisync \
          -Wl,--as-needed"
CUDNN_LIB=/usr/lib/aarch64-linux-gnu
aarch64-linux-gnu-gcc -O2 \
    $OUT/main.o $OUT/wrapper.o $OUT/kernel.o $OUT/rt_cuda.o $OUT/rt_pva.o \
    -L$CUDA/lib -L$CUDA/lib/stubs -L$CUDNN_LIB \
    $PVA_LINK \
    -lcudnn -lcublasLt -lcublas -lcudart -lm -lpthread -ldl -lstdc++ \
    -Wl,--allow-shlib-undefined \
    -Wl,-rpath,/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/nvidia:${JET_PVA_LIB} \
    -o $OUT/bilateral_jetson

echo "[bilateral/$DTYPE/$SIZE] (6) link CPU-stub binary"
aarch64-linux-gnu-gcc -O2 \
    $OUT/main.o $OUT/wrapper.o $OUT/kernel.o $OUT/rt_cpu.o \
    -lm -lpthread -o $OUT/bilateral_jetson_cpustub

echo ""
echo "═══ boxfilter ${DTYPE} ${SIZE}×${SIZE} binaries ═══"
ls -la $OUT/bilateral_jetson $OUT/bilateral_jetson_cpustub
