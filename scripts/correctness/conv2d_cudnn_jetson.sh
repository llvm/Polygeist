#!/bin/bash
# conv2d_cudnn_jetson.sh — cross-build extracted conv2d for Jetson Orin
# with the matched kernel.launch → cudnnConvolutionForward routing.
#
# Usage: ./conv2d_cudnn_jetson.sh [SIZE]   (default 256; baked via -DNI/-DNJ)
# Output: /tmp/conv2d_jetson_<SIZE>/{conv2d_jetson, conv2d_jetson_cpustub}

set -euo pipefail
source /home/arjaiswal/Polygeist/envsetup.sh

SIZE=${1:-256}
SCRIPTS=/home/arjaiswal/Polygeist/scripts/correctness
RT=/home/arjaiswal/Polygeist/runtime
EXT=/home/arjaiswal/Polygeist/third_party/polybenchGpu-extracted
OUT=/tmp/conv2d_jetson_${SIZE}
mkdir -p $OUT
CUDA=/usr/local/cuda-12.6/targets/sbsa-linux
# cuDNN cross package installs to /usr/{include,lib}/aarch64-linux-gnu/
CUDNN_INC=/usr/include/aarch64-linux-gnu
CUDNN_LIB=/usr/lib/aarch64-linux-gnu

echo "[conv2d/$SIZE] (1) cgeist → affine MLIR"
cgeist $EXT/conv2d.c --function=kernel_conv2d --resource-dir=/usr/lib/clang/14 \
    -DNI=$SIZE -DNJ=$SIZE --raise-scf-to-affine -fPIC -S \
    -o $OUT/orig.mlir 2>/dev/null

echo "[conv2d/$SIZE] (2) raise + lower-submap"
polygeist-opt --select-func=func-name=kernel_conv2d \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    $OUT/orig.mlir -o $OUT/linalg.mlir 2>$OUT/raise.err

echo "[conv2d/$SIZE] (3) kernel-match"
PYTHON=/home/arjaiswal/slacker/.venv/bin/python3
$PYTHON $SCRIPTS/kernel_match_rewrite.py $OUT/linalg.mlir > $OUT/matched.mlir 2>$OUT/match.err
N_LAUNCH=$(grep -c '@cudnnConvolution2D_9tap' $OUT/matched.mlir || true)
[ "${N_LAUNCH:-0}" -ge 1 ] || { echo "  FAIL: matcher didn't emit conv2d launch"; exit 1; }
echo "  matched $N_LAUNCH conv2d_9tap launch(es)"

echo "[conv2d/$SIZE] (4) inject defn"
awk '/^module attributes/ && !done{
       print;
       print "  kernel.defn @cudnnConvolution2D_9tap(%a0: memref<?x?xf64, strided<[?, 1], offset: ?>>, %a1: memref<?x?xf64, strided<[?, 1], offset: ?>>, %a2: memref<?x?xf64, strided<[?, 1], offset: ?>>, %a3: memref<?x?xf64, strided<[?, 1], offset: ?>>, %a4: memref<?x?xf64, strided<[?, 1], offset: ?>>, %a5: memref<?x?xf64, strided<[?, 1], offset: ?>>, %a6: memref<?x?xf64, strided<[?, 1], offset: ?>>, %a7: memref<?x?xf64, strided<[?, 1], offset: ?>>, %a8: memref<?x?xf64, strided<[?, 1], offset: ?>>, %c: memref<?x?xf64, strided<[?, 1], offset: ?>>, %w0: f64, %w1: f64, %w2: f64, %w3: f64, %w4: f64, %w5: f64, %w6: f64, %w7: f64, %w8: f64) { kernel.yield }";
       done=1; next
     }{print}' $OUT/matched.mlir > $OUT/matched_with_defn.mlir

echo "[conv2d/$SIZE] (5) lower-kernel-launch-to-cublas"
polygeist-opt --lower-kernel-launch-to-cublas \
    $OUT/matched_with_defn.mlir -o $OUT/abi.mlir 2>$OUT/abi.err

echo "[conv2d/$SIZE] (6) lower to LLVM, translate, retarget aarch64"
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

echo "[conv2d/$SIZE] (7) cross-compile harness + wrapper + runtimes"
# -march=armv8.2-a+fp16+bf16: Jetson Orin (Cortex-A78AE) is ARMv8.2-A
# baseline; we add +fp16 + +bf16 to enable scalar _Float16 / __bf16 support
# in the runtime so the f16/bf16 conv shims compile. cuDNN itself handles
# the hardware-acceleration path on the GPU side.
ARCH_FLAGS="-march=armv8.2-a+fp16+bf16"
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS -DNI=$SIZE -DNJ=$SIZE -c $SCRIPTS/conv2d_main_harness.c -o $OUT/main.o
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS -c $SCRIPTS/conv2d_jetson_wrapper.c -o $OUT/wrapper.o
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS -I$CUDA/include -I$CUDNN_INC -c $RT/polygeist_cublas_rt_cuda.c -o $OUT/rt_cuda.o
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS -c $RT/polygeist_cublas_rt_cpu.c -o $OUT/rt_cpu.o

echo "[conv2d/$SIZE] (8) link CUDA binary"
aarch64-linux-gnu-gcc -O2 \
    $OUT/main.o $OUT/wrapper.o $OUT/kernel.o $OUT/rt_cuda.o \
    -L$CUDA/lib -L$CUDA/lib/stubs -L$CUDNN_LIB \
    -lcudnn -lcublas -lcudart -lm -lpthread -ldl \
    -Wl,-rpath,/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu \
    -o $OUT/conv2d_jetson

echo "[conv2d/$SIZE] (9) link CPU-stub binary"
aarch64-linux-gnu-gcc -O2 \
    $OUT/main.o $OUT/wrapper.o $OUT/kernel.o $OUT/rt_cpu.o \
    -lm -lpthread -o $OUT/conv2d_jetson_cpustub

echo ""
echo "═══ ${SIZE}×${SIZE} binaries ═══"
ls -la $OUT/conv2d_jetson $OUT/conv2d_jetson_cpustub
aarch64-linux-gnu-readelf -d $OUT/conv2d_jetson | grep -E 'libcudnn|libcublas|libcudart' | head -4
