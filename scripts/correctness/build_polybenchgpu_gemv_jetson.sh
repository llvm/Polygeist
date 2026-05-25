#!/bin/bash
# build_polybenchgpu_gemv_jetson.sh KERNEL DATASET
# Build a polybenchGpu gemv-based kernel (atax, bicg, mvt, gemver, gesummv) end-to-end for Jetson.
# Handles 2D memref<?xMxf64> + 1D memref<?xf64> shapes, multiple kernel.launch callees.
set -euo pipefail
source /home/arjaiswal/Polygeist/envsetup.sh

KERNEL=${1:?"need kernel: atax|bicg|mvt|gemver|gesummv"}
DATASET=${2:?"need dataset: MINI|LARGE|EXTRALARGE"}

PY=/home/arjaiswal/slacker/.venv/bin/python3
SCRIPTS=/home/arjaiswal/Polygeist/scripts/correctness

ROOT=/home/arjaiswal/Polygeist/third_party/polybenchGpu/OpenMP
UTIL=$ROOT/utilities
KDIR=$ROOT/linear-algebra/kernels/$KERNEL
SRC=$(ls $KDIR/*.c | head -1)
FN="kernel_${KERNEL}"

OUT=/tmp/${KERNEL}_pbgpu_jetson_build
mkdir -p $OUT

HARNESS_CFLAGS=(-O3 -I"$UTIL" -I"$KDIR"
                -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_DUMP_ARRAYS
                -D${DATASET}_DATASET -Dstatic= -DPOLYBENCH_USE_C99_PROTO)
CGEIST_FLAGS=(-I"$UTIL" -I"$KDIR" -DDATA_TYPE_IS_DOUBLE
              -D${DATASET}_DATASET -Dstatic=
              --resource-dir=/usr/lib/clang/14
              --raise-scf-to-affine -fPIC -S)

echo "[$KERNEL/$DATASET] (1) cgeist"
cgeist "$SRC" --function='*' --no-inline "${CGEIST_FLAGS[@]}" \
   -o $OUT/${DATASET}_affine.mlir 2>$OUT/${DATASET}.cgeist.err
[ -s $OUT/${DATASET}_affine.mlir ] || { echo "FAIL"; head -3 $OUT/${DATASET}.cgeist.err; exit 1; }

echo "[$KERNEL/$DATASET] (2) raise + debuf"
polygeist-opt --select-func="func-name=$FN" \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    --linalg-debufferize \
    $OUT/${DATASET}_affine.mlir -o $OUT/${DATASET}_debuf.mlir 2>$OUT/${DATASET}.raise.err

echo "[$KERNEL/$DATASET] (3) matcher"
$PY $SCRIPTS/kernel_match_rewrite.py $OUT/${DATASET}_debuf.mlir \
    > $OUT/${DATASET}_matched.mlir 2>$OUT/${DATASET}.match.err
N_LAUNCH=$(grep -c "kernel.launch" $OUT/${DATASET}_matched.mlir || true)
echo "    matched $N_LAUNCH kernel.launch ops"
[ "${N_LAUNCH:-0}" -ge 1 ] || { echo "matcher FAIL"; exit 1; }

echo "[$KERNEL/$DATASET] (4) inject kernel.defn for every distinct callee"
# Determine the 2D static second dim
SECOND_DIM=$(grep -oE "tensor<\?x[0-9]+xf64>" $OUT/${DATASET}_matched.mlir | head -1 | sed -E 's/tensor<\?x([0-9]+)xf64>/\1/')
echo "    static 2D dim: ${SECOND_DIM:-(none, 1D only)}"

$PY - <<EOF > $OUT/${DATASET}_matched_with_defn.mlir
import re
sec2d = "${SECOND_DIM:-}"
ty2d = f"tensor<?x{sec2d}xf64>" if sec2d else "tensor<?x?xf64>"
ty1d = "tensor<?xf64>"

callees = set()
with open("$OUT/${DATASET}_matched.mlir") as f:
    for line in f:
        m = re.search(r'kernel\.launch\s+@([A-Za-z0-9_]+)', line)
        if m: callees.add(m.group(1))

# Per-callee signature builders
def defn_for(name):
    if name == "cublasDgemv":
        return f"kernel.defn @{name}(%A: {ty2d}, %x: {ty1d}, %y: {ty1d}) -> {ty1d} {{ kernel.yield %y : {ty1d} }}"
    if name == "cublasDgemv_T":
        return f"kernel.defn @{name}(%A: {ty2d}, %x: {ty1d}, %y: {ty1d}) -> {ty1d} {{ kernel.yield %y : {ty1d} }}"
    if name == "cublasDgemv_alpha":
        return f"kernel.defn @{name}(%A: {ty2d}, %x: {ty1d}, %y: {ty1d}, %alpha: f64) -> {ty1d} {{ kernel.yield %y : {ty1d} }}"
    if name == "cublasDaxpby":
        return f"kernel.defn @{name}(%x: {ty1d}, %y: {ty1d}, %alpha: f64, %beta: f64) -> {ty1d} {{ kernel.yield %y : {ty1d} }}"
    if name == "cublasDaxpy_unit":
        return f"kernel.defn @{name}(%x: {ty1d}, %y: {ty1d}) -> {ty1d} {{ kernel.yield %y : {ty1d} }}"
    if name == "cublasDger_rank2":
        return f"kernel.defn @{name}(%u1: {ty1d}, %v1: {ty1d}, %u2: {ty1d}, %v2: {ty1d}, %A: {ty2d}) -> {ty2d} {{ kernel.yield %A : {ty2d} }}"
    if name == "memset_zero_1D":
        return f"kernel.defn @{name}(%v: {ty1d}) -> {ty1d} {{ kernel.yield %v : {ty1d} }}"
    if name == "cublasDgemm":
        return f"kernel.defn @{name}(%A: {ty2d}, %B: {ty2d}, %C: {ty2d}, %beta: f64, %alpha: f64) -> {ty2d} {{ kernel.yield %C : {ty2d} }}"
    if name == "cublasDgemm_simple":
        return f"kernel.defn @{name}(%A: {ty2d}, %B: {ty2d}, %C: {ty2d}) -> {ty2d} {{ kernel.yield %C : {ty2d} }}"
    if name == "cublasDgemm_alpha_only":
        return f"kernel.defn @{name}(%A: {ty2d}, %B: {ty2d}, %C: {ty2d}, %alpha: f64) -> {ty2d} {{ kernel.yield %C : {ty2d} }}"
    if name == "cublasDgeam_scale2D":
        return f"kernel.defn @{name}(%M: {ty2d}, %s: f64) -> {ty2d} {{ kernel.yield %M : {ty2d} }}"
    if name == "memset_zero_2D":
        return f"kernel.defn @{name}(%M: {ty2d}) -> {ty2d} {{ kernel.yield %M : {ty2d} }}"
    raise SystemExit(f"unknown callee in matched MLIR: {name}")

done = False
with open("$OUT/${DATASET}_matched.mlir") as f:
    for line in f:
        print(line, end='')
        if not done and line.startswith("module attributes"):
            for c in sorted(callees):
                print("  " + defn_for(c))
            done = True
EOF
sed -i 's/!any/f64/g' $OUT/${DATASET}_matched_with_defn.mlir

echo "[$KERNEL/$DATASET] (5) lower-kernel-launch-to-cublas"
polygeist-opt --lower-kernel-launch-to-cublas \
    $OUT/${DATASET}_matched_with_defn.mlir -o $OUT/${DATASET}_abi.mlir 2>$OUT/${DATASET}.abi.err
[ -s $OUT/${DATASET}_abi.mlir ] || { echo "ABI FAIL"; head -5 $OUT/${DATASET}.abi.err; exit 1; }

# Rename + de-internal
sed -i "s/@${FN}\b/@${FN}_impl/g; s/llvm.linkage = #llvm.linkage<internal>//; s/func.func private @${FN}_impl/func.func @${FN}_impl/" \
    $OUT/${DATASET}_abi.mlir

echo "[$KERNEL/$DATASET] (6) build_jetson.sh → aarch64 binary"
aarch64-linux-gnu-gcc "${HARNESS_CFLAGS[@]}" -c "$SRC" -o $OUT/${DATASET}_full.o
aarch64-linux-gnu-objcopy --weaken-symbol=$FN $OUT/${DATASET}_full.o $OUT/${DATASET}_nokernel.o
aarch64-linux-gnu-gcc "${HARNESS_CFLAGS[@]}" -c "$UTIL/polybench.c" -o $OUT/${DATASET}_polybench.o

bash $SCRIPTS/build_jetson.sh \
    $OUT/${DATASET}_abi.mlir \
    $OUT/${KERNEL}_jetson_${DATASET} \
    $SCRIPTS/${KERNEL}_jetson_wrapper.c \
    $OUT/${DATASET}_nokernel.o \
    $OUT/${DATASET}_polybench.o 2>&1 | tail -3
echo "OK: $OUT/${KERNEL}_jetson_${DATASET}"
