#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
torch_root=/tmp/aten_native_sysroot_24_12/usr/local/lib/python3.12/dist-packages/torch
cuda_root=/usr/local/cuda-12.6/targets/sbsa-linux
aarch64-linux-gnu-g++ -std=c++17 -O2 -DHAVE_SVE256_CPU_DEFINITION \
  -I"$torch_root/include" -I"$torch_root/include/torch/csrc/api/include" \
  -I"$cuda_root/include" -c "$here/native_bridge.cpp" -o "$here/native_bridge.o"
python3 "$here/build_comparison.py"
