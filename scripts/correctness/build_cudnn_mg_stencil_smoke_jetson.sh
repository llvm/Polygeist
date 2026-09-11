#!/bin/bash
# Cross-compile the external cuDNN MG-stencil smoke for Jetson.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

OUT="${1:-/tmp/cudnn_mg_stencil_smoke_jetson}"
CUDA_CROSS="${CUDA_CROSS:-/usr/local/cuda-12.6/targets/sbsa-linux}"
CC="${AARCH64_CC:-aarch64-linux-gnu-gcc}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"$CC" -O2 -ffunction-sections -fdata-sections \
  -I"$RT" -I"$CUDA_CROSS/include" -I/usr/include/aarch64-linux-gnu \
  -c "$RT/polygeist_cublas_rt_cuda.c" -o "$WORK/runtime.o"
"$CC" -O2 -I"$RT" \
  -c "$REPO_ROOT/issues/ginsbach_asplos18/cudnn_mg_stencil_smoke.c" \
  -o "$WORK/smoke.o"
"$CC" -O2 -Wl,--gc-sections -o "$OUT" "$WORK/smoke.o" "$WORK/runtime.o" \
  -L"$CUDA_CROSS/lib" -L"$CUDA_CROSS/lib/stubs" \
  -L/usr/lib/aarch64-linux-gnu \
  -lcudnn -lcusparse -lcublasLt -lcublas -lcufft -lcusolver -lcudart \
  -lm -lpthread -ldl -Wl,--allow-shlib-undefined \
  -Wl,-rpath,/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu
file "$OUT"
