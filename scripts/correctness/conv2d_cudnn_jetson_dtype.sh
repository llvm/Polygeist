#!/bin/bash
# conv2d_cudnn_jetson_dtype.sh — cross-build extracted conv2d_<DTYPE>.c for
# Jetson Orin with the matched kernel.launch → cudnnConvolutionForward
# routing. Generalises conv2d_cudnn_jetson.sh to all dtypes in the Phase-2
# matrix (f64/f32/f16/bf16/i32/i16).
#
# Usage: ./conv2d_cudnn_jetson_dtype.sh <DTYPE> [SIZE]
#   <DTYPE>: f64 | f32 | f16 | bf16 | i32 | i16
#   [SIZE]:  default 256
#
# Output: /tmp/conv2d_jetson_<DTYPE>_<SIZE>/{conv2d_jetson,
#                                              conv2d_jetson_cpustub}

set -euo pipefail
source /home/arjaiswal/Polygeist/envsetup.sh

DTYPE=${1:?"missing DTYPE arg (f64|f32|f16|bf16|i32|i16)"}
SIZE=${2:-256}
SCRIPTS=/home/arjaiswal/Polygeist/scripts/correctness
RT=/home/arjaiswal/Polygeist/runtime
EXT=/home/arjaiswal/Polygeist/third_party/polybenchGpu-extracted
OUT=/tmp/conv2d_jetson_${DTYPE}_${SIZE}
mkdir -p $OUT
CUDA=/usr/local/cuda-12.6/targets/sbsa-linux
CUDNN_INC=/usr/include/aarch64-linux-gnu
CUDNN_LIB=/usr/lib/aarch64-linux-gnu

# Per-dtype config: source-file suffix, MLIR/MLIR-defn elem type, C scalar
# type, printf format. The kernel.launch symbol gets the dtype suffix; f64
# has no suffix for backward compat with the original Lit-surfacing test.
case "$DTYPE" in
  f64)  SRC=$EXT/conv2d.c;       MTY=f64;  CTY=double; KIND_DEF="-DCTYPE_KIND_FLOAT"; SYM_SUFFIX="";    ;;
  f32)  SRC=$EXT/conv2d_f32.c;   MTY=f32;  CTY=float;  KIND_DEF="-DCTYPE_KIND_FLOAT"; SYM_SUFFIX="_f32";;
  i32)  SRC=$EXT/conv2d_i32.c;   MTY=i32;  CTY=int;    KIND_DEF="-DCTYPE_KIND_INT";   SYM_SUFFIX="_i32";;
  i16)  SRC=$EXT/conv2d_i16.c;   MTY=i16;  CTY=short;  KIND_DEF="-DCTYPE_KIND_INT";   SYM_SUFFIX="_i16";;
  f16)
    echo "f16 not yet supported via cgeist (BuiltinType _Float16 unhandled in clang-mlir.cc)"; exit 2;;
  bf16)
    echo "bf16 not yet supported via cgeist"; exit 2;;
  *) echo "unknown dtype: $DTYPE"; exit 1;;
esac

[ -f "$SRC" ] || { echo "missing source $SRC"; exit 1; }

echo "[conv2d/$DTYPE/$SIZE] (1) cgeist → affine MLIR"
cgeist $SRC --function=kernel_conv2d --resource-dir=/usr/lib/clang/14 \
    -DNI=$SIZE -DNJ=$SIZE --raise-scf-to-affine -fPIC -S \
    -o $OUT/orig.mlir 2>/dev/null

echo "[conv2d/$DTYPE/$SIZE] (2) raise + lower-submap"
polygeist-opt --select-func=func-name=kernel_conv2d \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    $OUT/orig.mlir -o $OUT/linalg.mlir 2>$OUT/raise.err

echo "[conv2d/$DTYPE/$SIZE] (3) kernel-match"
PYTHON=/home/arjaiswal/slacker/.venv/bin/python3
$PYTHON $SCRIPTS/kernel_match_rewrite.py $OUT/linalg.mlir > $OUT/matched.mlir 2>$OUT/match.err
SYM="@cudnnConvolution2D_9tap${SYM_SUFFIX}"
N_LAUNCH=$(grep -c "$SYM" $OUT/matched.mlir || true)
[ "${N_LAUNCH:-0}" -ge 1 ] || { echo "  FAIL: matcher didn't emit $SYM launch"; exit 1; }
echo "  matched $N_LAUNCH ${SYM} launch(es)"

echo "[conv2d/$DTYPE/$SIZE] (4) inject dtype defn"
awk -v mty=$MTY -v sfx=$SYM_SUFFIX '/^module/ && !done{
       print;
       printf "  kernel.defn @cudnnConvolution2D_9tap%s(", sfx;
       for (k=0;k<10;k++) {
         printf "%%a%d: memref<?x?x%s, strided<[?, 1], offset: ?>>%s", k, mty, (k<9?", ":"");
       }
       printf ", ";
       for (k=0;k<9;k++) {
         printf "%%w%d: %s%s", k, mty, (k<8?", ":"");
       }
       print ") { kernel.yield }";
       done=1; next
     }{print}' $OUT/matched.mlir > $OUT/matched_with_defn.mlir

echo "[conv2d/$DTYPE/$SIZE] (5) lower-kernel-launch-to-cublas"
polygeist-opt --lower-kernel-launch-to-cublas \
    $OUT/matched_with_defn.mlir -o $OUT/abi.mlir 2>$OUT/abi.err

echo "[conv2d/$DTYPE/$SIZE] (6) lower to LLVM, translate, retarget aarch64"
MLIR_OPT=/home/arjaiswal/Polygeist/llvm-project/build/bin/mlir-opt
MLIR_TRANSLATE=/home/arjaiswal/Polygeist/llvm-project/build/bin/mlir-translate
CLANG=/home/arjaiswal/Polygeist/llvm-project/build/bin/clang
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

echo "[conv2d/$DTYPE/$SIZE] (7) cross-compile harness + wrapper + runtimes"
ARCH_FLAGS="-march=armv8.2-a+fp16+bf16"
DEFS="-DNI=$SIZE -DNJ=$SIZE -DCTYPE=$CTY $KIND_DEF"
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS $DEFS -c $SCRIPTS/conv2d_main_harness_dtype.c -o $OUT/main.o
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS -DCTYPE=$CTY -c $SCRIPTS/conv2d_jetson_wrapper_dtype.c -o $OUT/wrapper.o
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS -I$CUDA/include -I$CUDNN_INC -c $RT/polygeist_cublas_rt_cuda.c -o $OUT/rt_cuda.o
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS -c $RT/polygeist_cublas_rt_cpu.c -o $OUT/rt_cpu.o

echo "[conv2d/$DTYPE/$SIZE] (8) link CUDA binary"
aarch64-linux-gnu-gcc -O2 \
    $OUT/main.o $OUT/wrapper.o $OUT/kernel.o $OUT/rt_cuda.o \
    -L$CUDA/lib -L$CUDA/lib/stubs -L$CUDNN_LIB \
    -lcudnn -lcublas -lcudart -lm -lpthread -ldl \
    -Wl,-rpath,/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu \
    -o $OUT/conv2d_jetson

echo "[conv2d/$DTYPE/$SIZE] (9) link CPU-stub binary"
aarch64-linux-gnu-gcc -O2 \
    $OUT/main.o $OUT/wrapper.o $OUT/kernel.o $OUT/rt_cpu.o \
    -lm -lpthread -o $OUT/conv2d_jetson_cpustub

echo ""
echo "═══ ${DTYPE} ${SIZE}×${SIZE} binaries ═══"
ls -la $OUT/conv2d_jetson $OUT/conv2d_jetson_cpustub
