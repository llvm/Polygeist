#!/bin/bash
# polybench_cublas_jetson.sh — generic polybench → Jetson cross-build wrapper.
# Generalises gemm_cublas_jetson.sh to any polybench kernel whose body lifts
# to a matched kernel.launch @cublasDgemm op.
#
# Usage:
#   ./polybench_cublas_jetson.sh <kernel> [DATASET]
#
# Currently registered kernels (extend the KERNELS table below):
#   gemm, 2mm, 3mm
#
# DATASET defaults to LARGE. Allowed: MINI|SMALL|MEDIUM|LARGE|EXTRALARGE.
# (PolyBench/C 4.2.1 doesn't have STANDARD; passing it is a silent no-op.)

set -euo pipefail
source /home/arjaiswal/Polygeist/envsetup.sh

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <kernel> [DATASET]" >&2
  echo "       supported kernels: gemm, 2mm, 3mm" >&2
  exit 1
fi

KERNEL=$1
DATASET=${2:-LARGE}

case "$DATASET" in
  MINI|SMALL|MEDIUM|LARGE|EXTRALARGE) ;;
  STANDARD) echo "ERROR: PolyBench/C 4.2.1 has no STANDARD_DATASET (no-op). Use LARGE." >&2; exit 1 ;;
  *) echo "ERROR: bad DATASET '$DATASET'" >&2; exit 1 ;;
esac

POLYBENCH_DIR=/home/arjaiswal/Polygeist/tools/cgeist/Test/polybench
case "$KERNEL" in
  gemm) SRC_DIR="$POLYBENCH_DIR/linear-algebra/blas/gemm";     KFN=kernel_gemm ;;
  2mm)  SRC_DIR="$POLYBENCH_DIR/linear-algebra/kernels/2mm";   KFN=kernel_2mm ;;
  3mm)  SRC_DIR="$POLYBENCH_DIR/linear-algebra/kernels/3mm";   KFN=kernel_3mm ;;
  *)    echo "ERROR: kernel '$KERNEL' not registered in $0" >&2; exit 1 ;;
esac

UTIL=$POLYBENCH_DIR/utilities
SCRIPTS=/home/arjaiswal/Polygeist/scripts/correctness
RT=/home/arjaiswal/Polygeist/runtime
OUT=/tmp/polybench_jetson_${KERNEL}_${DATASET}
mkdir -p $OUT

WRAPPER=$SCRIPTS/${KERNEL}_jetson_wrapper.c
[ -f "$WRAPPER" ] || { echo "ERROR: wrapper missing at $WRAPPER" >&2; exit 1; }

CFLAGS=(-O3 -I"$UTIL" -I"$SRC_DIR"
        -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_DUMP_ARRAYS
        -D${DATASET}_DATASET
        -Dstatic= -DPOLYBENCH_USE_C99_PROTO)

echo "[$KERNEL/$DATASET] (1) cgeist → affine MLIR"
cgeist "$SRC_DIR/${KERNEL}.c" --function=$KFN --resource-dir=/usr/lib/clang/14 \
    "${CFLAGS[@]}" --raise-scf-to-affine -S \
    -o $OUT/orig.mlir 2>/dev/null

echo "[$KERNEL/$DATASET] (2) raise + lower-submap + debufferize"
polygeist-opt --select-func=func-name=$KFN \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    --linalg-debufferize \
    $OUT/orig.mlir -o $OUT/debuf.mlir 2>$OUT/raise.err

echo "[$KERNEL/$DATASET] (3) kernel-match"
PYTHON=/home/arjaiswal/slacker/.venv/bin/python3
$PYTHON $SCRIPTS/kernel_match_rewrite.py $OUT/debuf.mlir > $OUT/matched.mlir 2>$OUT/match.err
N_LAUNCH=$(grep -c '= kernel\.launch ' $OUT/matched.mlir || true)
N_LAUNCH=${N_LAUNCH:-0}
[ "$N_LAUNCH" -ge 1 ] || { echo "  FAIL: no kernel.launch ops"; exit 1; }
echo "  matched $N_LAUNCH kernel.launch op(s)"

echo "[$KERNEL/$DATASET] (4) inject kernel.defn declarations for all matched libsyms"
# The verifier requires every @<sym> referenced by a kernel.launch to have
# a kernel.defn @<sym> in scope. Inject stub defns for every library
# symbol our matcher emits; --lower-kernel-launch-to-cublas will clean
# them up after rewriting all launches into func.call ops.
awk '/^module attributes/ && !done{
       print;
       print "  kernel.defn @cublasDgemm(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>, %C: tensor<?x?xf64>, %beta: f64, %alpha: f64) -> tensor<?x?xf64> {";
       print "    kernel.yield %C : tensor<?x?xf64>";
       print "  }";
       print "  kernel.defn @cublasDgemm_simple(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>, %C: tensor<?x?xf64>) -> tensor<?x?xf64> {";
       print "    kernel.yield %C : tensor<?x?xf64>";
       print "  }";
       print "  kernel.defn @cublasDgemm_alpha_only(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>, %C: tensor<?x?xf64>, %alpha: f64) -> tensor<?x?xf64> {";
       print "    kernel.yield %C : tensor<?x?xf64>";
       print "  }";
       print "  kernel.defn @cublasDgeam_scale2D(%M: tensor<?x?xf64>, %scale: f64) -> tensor<?x?xf64> {";
       print "    kernel.yield %M : tensor<?x?xf64>";
       print "  }";
       print "  kernel.defn @memset_zero_2D(%M: tensor<?x?xf64>) -> tensor<?x?xf64> {";
       print "    kernel.yield %M : tensor<?x?xf64>";
       print "  }";
       done=1; next
     }{print}' $OUT/matched.mlir > $OUT/matched_with_defn.mlir

echo "[$KERNEL/$DATASET] (5) lower-kernel-launch-to-cublas"
polygeist-opt --lower-kernel-launch-to-cublas \
    $OUT/matched_with_defn.mlir -o $OUT/abi.mlir 2>$OUT/abi.err
N_CALL=$(grep -cE 'call @polygeist_cublas_dgemm\(' $OUT/abi.mlir || true)
N_CALL=${N_CALL:-0}
echo "  emitted $N_CALL func.call to polygeist_cublas_dgemm"

echo "[$KERNEL/$DATASET] (6) cross-compile polybench harness for aarch64"
aarch64-linux-gnu-gcc "${CFLAGS[@]}" -c "$SRC_DIR/${KERNEL}.c" -o $OUT/full.o
aarch64-linux-gnu-objcopy --weaken-symbol=$KFN $OUT/full.o $OUT/nokernel.o
aarch64-linux-gnu-gcc "${CFLAGS[@]}" -c "$UTIL/polybench.c" -o $OUT/polybench.o

echo "[$KERNEL/$DATASET] (7) rename @${KFN} → @${KFN}_impl + build both variants"
sed "s/@${KFN}\\b/@${KFN}_impl/g" $OUT/abi.mlir > $OUT/abi_renamed.mlir

# build_jetson.sh's own sed for @kernel_gemm is a no-op for other kernels.
# It also expects a particular WORK layout, so for non-gemm kernels we do
# the cross-link manually to avoid name conflicts.
WORK=$OUT/work; mkdir -p $WORK
CUDA=/usr/local/cuda-12.6/targets/sbsa-linux

sed 's|bufferization\.to_tensor \(%[^ ]*\) :|bufferization.to_tensor \1 restrict :|g' \
    $OUT/abi_renamed.mlir > $WORK/abi.mlir
/home/arjaiswal/Polygeist/llvm-project/build/bin/mlir-opt \
    --one-shot-bufferize=bufferize-function-boundaries \
    --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
    --convert-arith-to-llvm --finalize-memref-to-llvm \
    --convert-func-to-llvm --reconcile-unrealized-casts \
    $WORK/abi.mlir -o $WORK/llvm.mlir 2>&1 | tail -1
/home/arjaiswal/Polygeist/llvm-project/build/bin/mlir-translate \
    --mlir-to-llvmir $WORK/llvm.mlir -o $WORK/kernel.ll
sed -i 's|target triple = "x86_64.*"|target triple = "aarch64-linux-gnu"|;
        /^target datalayout/d' $WORK/kernel.ll
/home/arjaiswal/Polygeist/llvm-project/build/bin/clang \
    --target=aarch64-linux-gnu --gcc-toolchain=/usr \
    -O3 -c $WORK/kernel.ll -o $WORK/kernel.o 2>&1 | tail -1

# CUDA variant
aarch64-linux-gnu-gcc -O3 -I$CUDA/include -c $RT/polygeist_cublas_rt_cuda.c -o $WORK/rt_cuda.o
aarch64-linux-gnu-gcc -O3 -c $WRAPPER -o $WORK/wrapper.o
aarch64-linux-gnu-gcc -O2 \
    $OUT/nokernel.o $WORK/wrapper.o $WORK/kernel.o $WORK/rt_cuda.o $OUT/polybench.o \
    -L$CUDA/lib -L$CUDA/lib/stubs \
    -lcublas -lcudart -lm -lpthread -ldl \
    -Wl,-rpath,/usr/local/cuda/lib64 \
    -o $OUT/${KERNEL}_jetson

# CPU-stub variant
aarch64-linux-gnu-gcc -O3 -c $RT/polygeist_cublas_rt_cpu.c -o $WORK/rt_cpu.o
aarch64-linux-gnu-gcc -O2 \
    $OUT/nokernel.o $WORK/wrapper.o $WORK/kernel.o $WORK/rt_cpu.o $OUT/polybench.o \
    -lm -lpthread -o $OUT/${KERNEL}_jetson_cpustub

echo ""
echo "═══ ${KERNEL}/${DATASET} built for Jetson: ═══"
ls -la $OUT/${KERNEL}_jetson $OUT/${KERNEL}_jetson_cpustub
file $OUT/${KERNEL}_jetson | head -1
aarch64-linux-gnu-readelf -d $OUT/${KERNEL}_jetson | grep -E 'libcublas|libcudart' | head -3
