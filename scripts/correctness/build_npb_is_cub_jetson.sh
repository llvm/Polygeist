#!/bin/bash
# Cross-build NPB3.3-SER-C IS Class S with its source-faithful rank core
# raised by Polygeist.  The two zeroed integer histograms dispatch through
# CUB; the original driver and full_verify routine remain intact.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

OUT="${1:-$PWD/npb_is_cub_jetson}"
CUB_OUT="$(dirname "$OUT")/libpolygeist_cub.so"
NPB_ROOT="$REPO_ROOT/third_party/ginsbach-snu-npb/NPB3.3-SER-C"
IS_SOURCE="$NPB_ROOT/IS/is.c"
COMMON="$NPB_ROOT/common"
CORE="$REPO_ROOT/issues/ginsbach_asplos18/extracted/npb_is_rank_core.c"
ADAPTER="$REPO_ROOT/issues/ginsbach_asplos18/npb_is_cub_adapter.c"
PARAMS="$REPO_ROOT/issues/ginsbach_asplos18/npb_is_class_s"
LLVM_BIN="$REPO_ROOT/llvm-project/build/bin"
CUDA_CROSS="${CUDA_CROSS:-/usr/local/cuda-12.6/targets/sbsa-linux}"
CUDNN_CROSS_LIB="${CUDNN_CROSS_LIB:-/usr/lib/aarch64-linux-gnu}"
NVCC="${NVCC:-/usr/local/cuda-12.6/bin/nvcc}"
CC="${AARCH64_CC:-aarch64-linux-gnu-gcc}"
CXX="${AARCH64_CXX:-aarch64-linux-gnu-g++}"

for required in "$IS_SOURCE" "$CORE" "$ADAPTER" "$PARAMS/npbparams.h"; do
  [ -f "$required" ] || {
    echo "ERROR: required input is missing: $required" >&2
    exit 1
  }
done

WORK="$(mktemp -d)"
if [ "${POLYGEIST_KEEP_WORK:-0}" != "0" ]; then
  echo "[npb-is] keeping workdir: $WORK"
else
  trap 'rm -rf "$WORK"' EXIT
fi
mkdir -p "$WORK/lifted"

echo "[npb-is] raising rank core and selecting two CUB histograms"
PYTHON="${PYTHON:-/usr/bin/python3}" \
POLYGEIST_WRAP_KERNEL_PIPELINE=0 \
POLYGEIST_EXPORT_OBJECT_DIR="$WORK/lifted" \
POLYGEIST_SKIP_LINK=1 \
  "$SCRIPT_DIR/polygeist_build.sh" \
    --target=jetson --function=npb_is_rank_core \
    --harness="$CORE" -o "$WORK/unused" "$CORE"

HISTOGRAM_CALLS="$(grep -c \
  'call @polygeist_cub_histogram_even_i32_shift_zero' \
  "$WORK/lifted/abi.mlir" || true)"
if [ "$HISTOGRAM_CALLS" -ne 2 ]; then
  echo "ERROR: expected 2 static CUB histogram calls; got $HISTOGRAM_CALLS" >&2
  exit 1
fi

echo "[npb-is] preserving the original driver and replacing only rank"
"$LLVM_BIN/clang" --target=aarch64-linux-gnu --gcc-toolchain=/usr \
  -O0 -fno-inline -emit-llvm -c -I"$PARAMS" -I"$COMMON" \
  "$IS_SOURCE" -o "$WORK/is.bc"
"$LLVM_BIN/llvm-extract" --delete --func=rank \
  "$WORK/is.bc" -o "$WORK/is_without_rank.bc"
"$LLVM_BIN/clang" --target=aarch64-linux-gnu --gcc-toolchain=/usr \
  -O2 -ffunction-sections -fdata-sections \
  -c "$WORK/is_without_rank.bc" -o "$WORK/is.o"

"$CC" -O2 -ffunction-sections -fdata-sections -I"$PARAMS" \
  -c "$ADAPTER" -o "$WORK/adapter.o"
for source in c_print_results c_timers wtime; do
  "$CC" -O2 -ffunction-sections -fdata-sections -I"$COMMON" \
    -c "$COMMON/$source.c" -o "$WORK/$source.o"
done

echo "[npb-is] building external CUB companion library"
"$NVCC" -O3 -std=c++17 -arch=sm_87 -ccbin "$CXX" \
  -Xcompiler=-fPIC -shared "$REPO_ROOT/runtime/polygeist_cub_rt.cu" \
  -o "$CUB_OUT"

echo "[npb-is] linking AArch64 Class S executable: $OUT"
"$CC" -O2 -Wl,--gc-sections -o "$OUT" \
  "$WORK/is.o" "$WORK/adapter.o" "$WORK/c_print_results.o" \
  "$WORK/c_timers.o" "$WORK/wtime.o" \
  "$WORK/lifted/kernel.o" "$WORK/lifted/wrapper.o" \
  "$WORK/lifted/rt.o" "$WORK/lifted/mlir_runner_utils.o" \
  -L"$CUDA_CROSS/lib" -L"$CUDA_CROSS/lib/stubs" -L"$CUDNN_CROSS_LIB" \
  -lcudnn -lcublasLt -lcublas -lcufft -lcusparse -lcusolver -lcudart \
  -lm -lpthread -ldl \
  -Wl,-rpath,/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu

echo "[npb-is] complete: 2 CUB histogram sites"
echo "         deploy both $OUT and $CUB_OUT"
file "$OUT" "$CUB_OUT"
