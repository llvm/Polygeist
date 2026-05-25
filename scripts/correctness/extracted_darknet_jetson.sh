#!/bin/bash
# extracted_darknet_jetson.sh — cross-build a single extracted-darknet
# kernel for Jetson Orin via the matched kernel.launch → cuDNN runtime
# pipeline.
#
# Usage:
#   ./extracted_darknet_jetson.sh <KERNEL> <DATASET>
# Where KERNEL is one of: conv2d_batched, maxpool_batched,
# batchnorm_batched, shortcut_batched. DATASET is MINI or LARGE.
#
# Output dir:  /tmp/extracted_darknet_<KERNEL>_<DATASET>/
#   - <kernel>_jetson         (aarch64 ELF, links libcudnn / libcublas / libcudart)
#   - <kernel>_jetson_cpustub (aarch64 ELF, CPU reference shim — no GPU)
# Both binaries take no args; they init their inputs internally, run the
# kernel once, print POLYGEIST_TIMING + CHECKSUM + DUMP_ARRAYS on stderr.

set -euo pipefail
source /home/arjaiswal/Polygeist/envsetup.sh

KERNEL="${1:-conv2d_batched}"
DATASET="${2:-MINI}"

case "$KERNEL" in
  conv2d_batched|maxpool_batched|batchnorm_batched|shortcut_batched|conv_bn_relu_batched|conv_bias_relu_add_batched|gemm_bias_relu|ata_gemm|conv1x1_batched) ;;
  *) echo "Unknown kernel '$KERNEL'. Choose from: conv2d_batched, maxpool_batched, batchnorm_batched, shortcut_batched, conv_bn_relu_batched, conv_bias_relu_add_batched, gemm_bias_relu, ata_gemm, conv1x1_batched" >&2; exit 2 ;;
esac
case "$DATASET" in MINI|LARGE) ;;
  *) echo "DATASET must be MINI or LARGE (got '$DATASET')" >&2; exit 2 ;;
esac

SCRIPTS=/home/arjaiswal/Polygeist/scripts/correctness
RT=/home/arjaiswal/Polygeist/runtime
EXT=/home/arjaiswal/Polygeist/third_party/cnn-extracted
OUT=/tmp/extracted_darknet_${KERNEL}_${DATASET}
mkdir -p $OUT

CUDA=/usr/local/cuda-12.6/targets/sbsa-linux
CUDNN_INC=/usr/include/aarch64-linux-gnu
CUDNN_LIB=/usr/lib/aarch64-linux-gnu

DEF=""
[ "$DATASET" = "LARGE" ] && DEF="-DLARGE_DATASET"
[ "$DATASET" = "MINI"  ] && DEF="-DMINI_DATASET"

KERN_FN="kernel_${KERNEL}"

echo "[$KERNEL/$DATASET] (1) cgeist → affine MLIR"
cgeist $EXT/${KERNEL}.c --function=$KERN_FN \
       --resource-dir=/usr/lib/clang/14 $DEF \
       --raise-scf-to-affine -fPIC -S \
       -o $OUT/orig.mlir 2>$OUT/cgeist.err

echo "[$KERNEL/$DATASET] (2) raise + debufferize"
polygeist-opt --select-func=func-name=$KERN_FN \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline \
    $OUT/orig.mlir 2>$OUT/raise.err |
polygeist-opt --linalg-debufferize -o $OUT/linalg.mlir 2>>$OUT/raise.err

echo "[$KERNEL/$DATASET] (3) kernel-match"
PYTHON=/home/arjaiswal/slacker/.venv/bin/python3
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
$PYTHON $SCRIPTS/kernel_match_rewrite.py $OUT/linalg.mlir > $OUT/matched.mlir 2>$OUT/match.err
N_LAUNCH=$(grep -c 'kernel.launch' $OUT/matched.mlir || true)
[ "${N_LAUNCH:-0}" -ge 1 ] || { echo "  FAIL: no matcher hits"; exit 1; }
echo "  matched $N_LAUNCH kernel.launch op(s)"

echo "[$KERNEL/$DATASET] (4) inject kernel.defn"
$PYTHON /tmp/cnn_mlir/inject_defns.py $OUT/matched.mlir $OUT/matched_with_defn.mlir

echo "[$KERNEL/$DATASET] (4b) cleanup orphan submapInverse"
$PYTHON /tmp/cnn_mlir/cleanup_orphans.py $OUT/matched_with_defn.mlir $OUT/cleaned.mlir

echo "[$KERNEL/$DATASET] (5) lower-kernel-launch-to-cublas"
polygeist-opt --lower-kernel-launch-to-cublas \
    $OUT/cleaned.mlir -o $OUT/abi.mlir 2>$OUT/abi.err

echo "[$KERNEL/$DATASET] (6) lower polygeist.submap + MLIR → LLVM IR, retarget aarch64"
MLIR_OPT=/home/arjaiswal/Polygeist/llvm-project/build/bin/mlir-opt
MLIR_TRANSLATE=/home/arjaiswal/Polygeist/llvm-project/build/bin/mlir-translate
CLANG=/home/arjaiswal/Polygeist/llvm-project/build/bin/clang
# After ABI lowering the launch is gone but residual polygeist.submap /
# submapInverse ops are still there (their results were rewired by the
# lowering helper, so they're now DCE-able pure ops). Run polygeist-opt
# with --canonicalize first so they vanish before mlir-opt sees them
# (mlir-opt doesn't know the polygeist dialect).
polygeist-opt --canonicalize --cse --lower-polygeist-submap --canonicalize --cse \
    $OUT/abi.mlir -o $OUT/abi_canon.mlir 2>>$OUT/abi.err
$MLIR_OPT --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
    --expand-strided-metadata \
    --convert-arith-to-llvm --finalize-memref-to-llvm \
    --convert-func-to-llvm --reconcile-unrealized-casts \
    $OUT/abi_canon.mlir -o $OUT/llvm.mlir 2>$OUT/mlir.err
$MLIR_TRANSLATE --mlir-to-llvmir $OUT/llvm.mlir -o $OUT/kernel.ll
sed -i 's|target triple = "x86_64.*"|target triple = "aarch64-linux-gnu"|;
        /^target datalayout/d;
        s/@'$KERN_FN'\b/@'$KERN_FN'_impl/g' $OUT/kernel.ll
$CLANG --target=aarch64-linux-gnu --gcc-toolchain=/usr \
    -O3 -c $OUT/kernel.ll -o $OUT/kernel.o 2>&1 | tail -3

echo "[$KERNEL/$DATASET] (7) harness + runtime"
ARCH_FLAGS="-march=armv8.2-a+fp16+bf16"
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS $DEF \
    -c $SCRIPTS/${KERNEL}_jetson_harness.c -o $OUT/main.o
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS -I$CUDA/include -I$CUDNN_INC \
    -c $RT/polygeist_cublas_rt_cuda.c -o $OUT/rt_cuda.o
aarch64-linux-gnu-gcc -O3 $ARCH_FLAGS \
    -c $RT/polygeist_cublas_rt_cpu.c -o $OUT/rt_cpu.o

echo "[$KERNEL/$DATASET] (8) link CUDA binary"
aarch64-linux-gnu-gcc -O2 \
    $OUT/main.o $OUT/kernel.o $OUT/rt_cuda.o \
    -L$CUDA/lib -L$CUDA/lib/stubs -L$CUDNN_LIB \
    -lcudnn -lcublasLt -lcublas -lcudart -lm -lpthread -ldl \
    -Wl,-rpath,/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu \
    -o $OUT/${KERNEL}_jetson

echo "[$KERNEL/$DATASET] (9) link CPU-stub binary"
aarch64-linux-gnu-gcc -O2 \
    $OUT/main.o $OUT/kernel.o $OUT/rt_cpu.o \
    -lm -lpthread -o $OUT/${KERNEL}_jetson_cpustub

echo ""
echo "═══ ${KERNEL} / ${DATASET} ═══"
ls -la $OUT/${KERNEL}_jetson $OUT/${KERNEL}_jetson_cpustub
aarch64-linux-gnu-readelf -d $OUT/${KERNEL}_jetson | grep -E 'libcudnn|libcublas|libcudart' | head -4
