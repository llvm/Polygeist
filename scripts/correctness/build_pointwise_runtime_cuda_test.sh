#!/usr/bin/env bash
# Build the CUDA runtime memory/capture regression. Run the result on a CUDA GPU.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${CC:=cc}"
: "${CUDA_TARGET_ROOT:=/usr/local/cuda}"
: "${CUDNN_INCLUDE:=/usr/include}"
: "${CUDNN_LIB:=/usr/lib}"
: "${CUTENSOR_ROOT:?Set CUTENSOR_ROOT to the cuTENSOR installation root}"
output=${1:?Usage: build_pointwise_runtime_cuda_test.sh OUTPUT_BINARY}
mkdir -p "$(dirname -- "$output")"
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
includes=(-I"$repo_root/runtime" -I"$CUDA_TARGET_ROOT/include" -I"$CUDNN_INCLUDE" -I"$CUTENSOR_ROOT/include")
"$CC" -O2 -ffunction-sections -fdata-sections -DPOLYGEIST_DISABLE_CUSPARSE \
  -DPOLYGEIST_DISABLE_CUSOLVER -DPOLYGEIST_ENABLE_CUTENSOR "${includes[@]}" \
  -c "$repo_root/runtime/polygeist_cublas_rt_cuda.c" -o "$build_dir/runtime.o"
"$CC" -O2 "${includes[@]}" \
  -c "$repo_root/scripts/correctness/tests/test_pointwise_runtime_cuda.c" -o "$build_dir/test.o"
"$CC" "$build_dir/test.o" "$build_dir/runtime.o" \
  -L"$CUDA_TARGET_ROOT/lib" -L"$CUDA_TARGET_ROOT/lib64" -L"$CUDA_TARGET_ROOT/lib/stubs" -L"$CUDNN_LIB" -L"$CUTENSOR_ROOT/lib" \
  -Wl,--gc-sections -Wl,--allow-shlib-undefined \
  -lcudnn -lcublas -lcublasLt -lcutensor -lcudart -lm -ldl -o "$output"
printf 'Built %s. Run it with CUDA/cuDNN/cuTENSOR on LD_LIBRARY_PATH.\n' "$output"
