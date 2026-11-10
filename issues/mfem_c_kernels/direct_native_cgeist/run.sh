#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/home/arjaiswal/Polygeist}
out=${OUT:-/tmp/mfem-direct-native-cgeist}
cuda=${CUDA_HOME:-/usr/local/cuda-12.6}
mfem_build=${MFEM_AARCH64_BUILD:-/home/arjaiswal/mfem-section42-aarch64-build4}
cgeist=${CGEIST:-$repo/build/bin/cgeist}
mlir_opt=${MLIR_OPT:-$repo/llvm-project/build/bin/mlir-opt}
resource_dir=${CLANG_RESOURCE_DIR:-$repo/llvm-project/build/lib/clang/18}
source_cpp=$repo/issues/mfem_c_kernels/benchmarks/mfem_native_cuda_publication_bench.cpp
source_cu=$out/mfem_native_cuda_publication_bench.cu

mkdir -p "$out/include"
ln -sf "$source_cpp" "$source_cu"

# Clang's CUDA runtime wrapper includes this optional cuRAND implementation
# header even though this MFEM operator does not use cuRAND. The installed
# cross toolkit does not ship it, so provide an intentionally empty include.
if [[ ! -e "$cuda/targets/sbsa-linux/include/curand_mtgp32_kernel.h" ]]; then
  printf '%s\n' \
    '#pragma once' \
    '/* Unused optional cuRAND header required by Clang CUDA wrapper. */' \
    > "$out/include/curand_mtgp32_kernel.h"
fi

common=(
  --cuda-gpu-arch=sm_87
  --cuda-path="$cuda"
  --resource-dir="$resource_dir"
  --std=c++17
  --function=*
  -DBENCH_MASS_2D
  -DMFEM_BENCH_NE=1024
  -DMFEM_CONFIG_FILE="\"$mfem_build/config/_config.hpp\""
  -I"$out/include"
  -I"$cuda/targets/sbsa-linux/include"
  -I"$mfem_build"
  -I"$repo/third_party/mfem"
  -I"$repo/issues/mfem_c_kernels/benchmarks"
)

"$cgeist" "$source_cu" "${common[@]}" -S \
  -o "$out/mass2-cuda.mlir" \
  > "$out/cgeist.stdout" 2> "$out/cgeist.stderr"

set +e
"$mlir_opt" "$out/mass2-cuda.mlir" -o /dev/null \
  > "$out/verify.stdout" 2> "$out/verify.stderr"
verify_rc=$?

"$cgeist" "$source_cu" "${common[@]}" \
  --cuda-lower --cpuify=distribute -emit-llvm-dialect \
  -output-intermediate-gpu=1 -emit-cuda -c \
  -o "$out/mass2-cgeist.o" \
  > "$out/integrated.stdout" 2> "$out/integrated.stderr"
integrated_rc=$?
set -e

ONLY=mass2 OUT="$out/native-build" \
  bash "$repo/scripts/correctness/build_mfem_native_cuda_campaign.sh"

printf 'cgeist_mlir_lines=%s\n' "$(wc -l < "$out/mass2-cuda.mlir")"
printf 'gpu_launch_count=%s\n' "$(grep -c 'gpu.launch' "$out/mass2-cuda.mlir" || true)"
printf 'mulf_count=%s\n' "$(grep -c 'arith.mulf' "$out/mass2-cuda.mlir" || true)"
printf 'addf_count=%s\n' "$(grep -c 'arith.addf' "$out/mass2-cuda.mlir" || true)"
printf 'mlir_verify_rc=%d\n' "$verify_rc"
printf 'integrated_cuda_compile_rc=%d\n' "$integrated_rc"
file "$out/native-build/bin/mass2_mfem_native_cuda"
sha256sum "$out/native-build/bin/mass2_mfem_native_cuda"

# The two failures are the currently documented result, so do not hide them
# by making this evidence-collection script itself fail after the native build.
