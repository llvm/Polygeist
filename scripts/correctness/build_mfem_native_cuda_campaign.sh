#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/home/arjaiswal/Polygeist}
cuda=${CUDA_HOME:-/usr/local/cuda-12.6}
mfem_build=${MFEM_AARCH64_BUILD:-/home/arjaiswal/mfem-section42-aarch64-build4}
out=${OUT:-/home/arjaiswal/mfem-ne1024-native}
nvcc="$cuda/bin/nvcc"
host_cxx=${AARCH64_CXX:-/usr/bin/aarch64-linux-gnu-g++}
src="$repo/issues/mfem_c_kernels/benchmarks/mfem_native_cuda_publication_bench.cpp"

mkdir -p "$out/bin" "$out/obj" "$out/logs"

kernels=(
  'mass2:BENCH_MASS_2D:pa' 'mass3:BENCH_MASS_3D:pa'
  'diff2:BENCH_DIFFUSION_2D:pa' 'diff3:BENCH_DIFFUSION_3D:pa'
  'conv2:BENCH_CONVECTION_2D:pa' 'conv3:BENCH_CONVECTION_3D:pa'
  'curl2:BENCH_CURLCURL_2D:pa' 'curl3:BENCH_CURLCURL_3D:pa'
  'div2:BENCH_DIVDIV_2D:pa' 'div3:BENCH_DIVDIV_3D:pa'
  'iv2:BENCH_INTERP_VALUE_2D:dfem' 'iv3:BENCH_INTERP_VALUE_3D:dfem'
  'av2:BENCH_INTEGRATE_VALUE_2D:dfem'
  'ig2:BENCH_INTERP_GRAD_2D:dfem' 'ig3:BENCH_INTERP_GRAD_3D:dfem'
  'ag2:BENCH_INTEGRATE_GRAD_2D:dfem' 'ag3:BENCH_INTEGRATE_GRAD_3D:dfem'
)

for spec in "${kernels[@]}"; do
  IFS=: read -r short macro family <<<"$spec"
  if [[ -n ${ONLY:-} && ",${ONLY}," != *",${short},"* ]]; then
    continue
  fi
  defs=(-D"$macro")
  [[ $family == dfem ]] && defs+=(-DBENCH_DFEM)
  echo "building $short ($macro, $family)"
  "$nvcc" -forward-unknown-to-host-compiler -ccbin="$host_cxx" \
    -target-dir sbsa-linux --expt-extended-lambda --expt-relaxed-constexpr \
    -O3 -std=c++17 -arch=sm_87 \
    -DMFEM_CONFIG_FILE="\"$mfem_build/config/_config.hpp\"" \
    -DMFEM_BENCH_NE=1024 "${defs[@]}" \
    -I"$mfem_build" -I"$repo/third_party/mfem" \
    -I"$repo/issues/mfem_c_kernels/benchmarks" \
    -x cu -c "$src" -o "$out/obj/$short.o" \
    >"$out/logs/$short.compile.log" 2>&1
  "$nvcc" -forward-unknown-to-host-compiler -ccbin="$host_cxx" \
    -target-dir sbsa-linux -arch=sm_87 "$out/obj/$short.o" \
    "$mfem_build/libmfem.a" -L"$cuda/targets/sbsa-linux/lib" \
    -lcudart -lpthread -ldl -lrt -o "$out/bin/${short}_mfem_native_cuda" \
    >>"$out/logs/$short.compile.log" 2>&1
done

file "$out"/bin/*_mfem_native_cuda
