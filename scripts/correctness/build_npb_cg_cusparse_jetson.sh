#!/bin/bash
# Cross-build the original NPB3.3-SER-C CG Class S application with its
# conj_grad computational routine raised by Polygeist and dispatched through
# external cuSPARSE/cuBLAS library calls. Nothing is compiled on the Jetson.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

OUT="${1:-$PWD/npb_cg_cusparse_jetson}"
NPB_ROOT="$REPO_ROOT/third_party/ginsbach-snu-npb/NPB3.3-SER-C"
CG_SOURCE="$NPB_ROOT/CG/cg.c"
COMMON="$NPB_ROOT/common"
CORE="$REPO_ROOT/issues/ginsbach_asplos18/extracted/npb_cg_conj_grad_core.c"
ADAPTER="$REPO_ROOT/issues/ginsbach_asplos18/npb_cg_cusparse_adapter.c"
PARAMS="$REPO_ROOT/issues/ginsbach_asplos18/npb_cg_class_s"
LLVM_BIN="$REPO_ROOT/llvm-project/build/bin"
CUDA_CROSS="${CUDA_CROSS:-/usr/local/cuda-12.6/targets/sbsa-linux}"
CUDNN_CROSS_LIB="${CUDNN_CROSS_LIB:-/usr/lib/aarch64-linux-gnu}"
CC="${AARCH64_CC:-aarch64-linux-gnu-gcc}"

for required in "$CG_SOURCE" "$CORE" "$ADAPTER" "$PARAMS/npbparams.h"; do
  [ -f "$required" ] || {
    echo "ERROR: required input is missing: $required" >&2
    exit 1
  }
done

WORK="$(mktemp -d)"
if [ "${POLYGEIST_KEEP_WORK:-0}" != "0" ]; then
  echo "[npb-cg] keeping workdir: $WORK"
else
  trap 'rm -rf "$WORK"' EXIT
fi
mkdir -p "$WORK/lifted"

echo "[npb-cg] raising conj_grad core and selecting external libraries"
PYTHON="${PYTHON:-/usr/bin/python3}" \
POLYGEIST_WRAP_KERNEL_PIPELINE=0 \
POLYGEIST_EXPORT_OBJECT_DIR="$WORK/lifted" \
POLYGEIST_SKIP_LINK=1 \
  "$SCRIPT_DIR/polygeist_build.sh" \
    --target=jetson --function=npb_cg_conj_grad_core \
    --harness="$CORE" -o "$WORK/unused" "$CORE"

SPMV_CALLS="$(grep -c 'call @polygeist_cusparse_spmv_csr_f64_sized' \
  "$WORK/lifted/abi.mlir" || true)"
DOT_CALLS="$(grep -c 'call @polygeist_cublas_dot_f64' \
  "$WORK/lifted/abi.mlir" || true)"
if [ "$SPMV_CALLS" -ne 2 ] || [ "$DOT_CALLS" -ne 1 ]; then
  echo "ERROR: expected 2 cuSPARSE SpMV and 1 cuBLAS dot static calls; got " \
       "$SPMV_CALLS and $DOT_CALLS" >&2
  exit 1
fi

echo "[npb-cg] preserving the original NPB driver and replacing only conj_grad"
"$LLVM_BIN/clang" --target=aarch64-linux-gnu --gcc-toolchain=/usr \
  -O0 -fno-inline -emit-llvm -c -I"$PARAMS" -I"$COMMON" \
  "$CG_SOURCE" -o "$WORK/cg.bc"
"$LLVM_BIN/llvm-extract" --delete --func=conj_grad \
  "$WORK/cg.bc" -o "$WORK/cg_without_conj_grad.bc"
"$LLVM_BIN/clang" --target=aarch64-linux-gnu --gcc-toolchain=/usr \
  -O2 -ffunction-sections -fdata-sections \
  -c "$WORK/cg_without_conj_grad.bc" -o "$WORK/cg.o"

"$CC" -O2 -ffunction-sections -fdata-sections -I"$PARAMS" \
  -c "$ADAPTER" -o "$WORK/adapter.o"
for source in print_results randdp c_timers wtime; do
  "$CC" -O2 -ffunction-sections -fdata-sections -I"$COMMON" \
    -c "$COMMON/$source.c" -o "$WORK/$source.o"
done

echo "[npb-cg] linking AArch64 Class S executable: $OUT"
"$CC" -O2 -Wl,--gc-sections -o "$OUT" \
  "$WORK/cg.o" "$WORK/adapter.o" "$WORK/print_results.o" \
  "$WORK/randdp.o" "$WORK/c_timers.o" "$WORK/wtime.o" \
  "$WORK/lifted/kernel.o" "$WORK/lifted/wrapper.o" \
  "$WORK/lifted/rt.o" "$WORK/lifted/mlir_runner_utils.o" \
  -L"$CUDA_CROSS/lib" -L"$CUDA_CROSS/lib/stubs" -L"$CUDNN_CROSS_LIB" \
  -lcudnn -lcublasLt -lcublas -lcufft -lcusparse -lcusolver -lcudart \
  -lm -lpthread -ldl \
  -Wl,-rpath,/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu

echo "[npb-cg] complete: 2 cuSPARSE SpMV sites, 1 cuBLAS dot site"
file "$OUT"
