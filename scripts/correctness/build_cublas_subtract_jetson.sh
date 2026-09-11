#!/usr/bin/env bash
# Cross-compile the subtract GEMV/GEMM numerical smoke on the host.  The
# resulting AArch64 binary is ready to copy to and execute on Jetson silicon.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

OUT="${1:-/tmp/cublas_subtract_jetson}"
CUDA_CROSS="${CUDA:-/usr/local/cuda-12.6/targets/sbsa-linux}"
CUDNN_INC="${CUDNN_INC:-/usr/include/aarch64-linux-gnu}"
CUDNN_LIB="${CUDNN_LIB:-/usr/lib/aarch64-linux-gnu}"
CC="${AARCH64_CC:-aarch64-linux-gnu-gcc}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"$CC" -O3 -ffunction-sections -fdata-sections \
  -Wno-deprecated-declarations -Wno-discarded-qualifiers \
  -I"$CUDA_CROSS/include" -I"$CUDNN_INC" \
  -c "$REPO_ROOT/runtime/polygeist_cublas_rt_cuda.c" -o "$WORK/runtime.o"
"$CC" -O3 -I"$CUDA_CROSS/include" \
  -c "$SCRIPT_DIR/cublas_subtract_validation.c" \
  -o "$WORK/validation.o"
"$CC" -O3 "$WORK/runtime.o" "$WORK/validation.o" \
  -L"$CUDA_CROSS/lib" -L"$CUDA_CROSS/lib/stubs" -L"$CUDNN_LIB" \
  -lcublasLt -lcublas -lcusparse -lcusolver -lcudart -lm -lpthread -ldl \
  -Wl,--gc-sections -Wl,-rpath-link,"$CUDA_CROSS/lib" \
  '-Wl,-rpath,$ORIGIN:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu' \
  -o "$OUT"
file "$OUT"
