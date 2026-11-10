#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/home/arjaiswal/Polygeist}
cuda=${CUDA_HOME:-/usr/local/cuda-12.6}
host_cxx=${AARCH64_CXX:-/usr/bin/aarch64-linux-gnu-g++}
out=${OUT:-/tmp/mfem-isolated-mass2}
source_file=$repo/issues/mfem_c_kernels/direct_native_cgeist/isolated_mass2_native.cu

mkdir -p "$out"
"$cuda/bin/nvcc" -forward-unknown-to-host-compiler \
  -ccbin="$host_cxx" -target-dir sbsa-linux \
  -O3 -DNDEBUG -std=c++17 -arch=sm_87 \
  "$source_file" -o "$out/isolated_mass2_native" \
  > "$out/compile.log" 2>&1

file "$out/isolated_mass2_native"
sha256sum "$source_file" "$out/isolated_mass2_native"
