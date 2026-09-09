#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
torch_root=${ATEN_NATIVE_TORCH_ROOT:-/tmp/aten_native_sysroot_24_12/usr/local/lib/python3.12/dist-packages/torch}
cuda_root=${ATEN_NATIVE_CUDA_ROOT:-/usr/local/cuda-12.6/targets/sbsa-linux}
output=${1:-/tmp/aten_native_exact_region_bench}
source_file="$repo_root/issues/aten_c_kernels/native_cuda_results/native_exact_region_bench.cpp"

test -f "$torch_root/include/ATen/ATen.h"
test -f "$torch_root/lib/libtorch_cuda.so"
test -f "$cuda_root/include/cuda_runtime_api.h"

aarch64-linux-gnu-g++ -std=c++17 -O2 -DHAVE_SVE256_CPU_DEFINITION \
  -I"$torch_root/include" \
  -I"$torch_root/include/torch/csrc/api/include" \
  -I"$cuda_root/include" \
  "$source_file" -o "$output" \
  -L"$torch_root/lib" -L"$cuda_root/lib" \
  -Wl,-rpath,/home/nvidia/polygeist_aten_native_24_12 \
  -Wl,-rpath,/usr/local/cuda/lib64 \
  -Wl,-rpath,/usr/lib/aarch64-linux-gnu \
  -Wl,--allow-shlib-undefined \
  -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 \
  -lcudart -lpthread -ldl

file "$output"
