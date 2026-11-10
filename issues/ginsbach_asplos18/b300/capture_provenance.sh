#!/usr/bin/env bash
set -euo pipefail

root=/mnt/forge1/arpitj
result="$root/results/ginsbach-linearalgebra-b300"
compiler_build="$root/build/ginsbach-linearalgebra"
bench_build="$root/build/ginsbach-linearalgebra-bench"

mkdir -p "$result/sources" "$result/generated-harnesses" "$result/build-logs"
cp "$root/sources/ginsbach-linearalgebra/gemm_bench.c" \
  "$root/sources/ginsbach-linearalgebra/spmv_csr_bench.c" \
  "$root/sources/ginsbach-linearalgebra/cuda_event_adapter.cu" \
  "$root/sources/ginsbach-linearalgebra/gemm_native_cublas.cu" \
  "$root/sources/ginsbach-linearalgebra/spmv_csr_native_cusparse.cu" \
  "$result/sources/"
cp "$compiler_build/lilac/gemm_cuda.cc" \
  "$compiler_build/lilac/spmv_csr_cuda.cc" \
  "$compiler_build/lilac/gemm_naive.cc" \
  "$compiler_build/lilac/spmv_csr_naive.cc" \
  "$result/generated-harnesses/"
cp "$root/logs/ginsbach-linearalgebra-configure.log" \
  "$root/logs/ginsbach-linearalgebra-build.log" \
  "$root/logs/ginsbach-lilac-regenerate.log" \
  "$root/logs/ginsbach-gemm-cuda-build.log" \
  "$root/logs/ginsbach-spmv-cuda-build.log" \
  "$root/logs/ginsbach-spmv-cuda13-build.log" \
  "$result/build-logs/"
cp "$bench_build"/replace-report-*.json "$result"/
cp "$bench_build"/replace-source-*.ll "$result"/
cp "$root/logs/ginsbach-linearalgebra-cg-s/run-naive.log" \
  "$result/npb-cg-class-s-raised-naive.log"
cp "$root/logs/ginsbach-linearalgebra-cg-s/run-cusparse.log" \
  "$result/npb-cg-class-s-raised-cusparse.log"
cp "$root/repos/ginsbach-snu-npb/NPB3.3-SER-C/CG/replace-report-cg.c.json" \
  "$result/npb-cg-replace-report.json"
cp "$root/patches/ginsbach-linearalgebra-cuda13.patch" "$result"/

{
  echo "utc=$(date -u +%FT%TZ)"
  echo "hostname=$(hostname)"
  echo "architecture=$(uname -m)"
  echo "cpu=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
  echo "cpu_affinity=$(taskset -pc $$ | sed 's/.*: //')"
  echo "llvm_commit=$(git -C "$root/repos/ginsbach-llvm-linearalgebra" rev-parse HEAD)"
  echo "clang_commit=$(git -C "$root/repos/ginsbach-clang-research" rev-parse HEAD)"
  "$compiler_build/bin/clang" --version | head -2
  /usr/bin/clang --version | head -1
  /usr/local/cuda-13.1/bin/nvcc --version | tail -1
  nvidia-smi --query-gpu=index,name,uuid,driver_version \
    --format=csv,noheader | head -1
  grep -E '^#define CUBLAS_VER_(MAJOR|MINOR|PATCH)' \
    /usr/local/cuda-13.1/include/cublas_api.h
  grep -E '^#define CUSPARSE_VER_(MAJOR|MINOR|PATCH)' \
    /usr/local/cuda-13.1/include/cusparse.h
} > "$result/provenance.txt"

nm "$bench_build/gemm_raised.o" | grep gemm_harness \
  > "$result/emitted_symbols.txt"
nm "$bench_build/spmv_csr_raised.o" | grep spmv_csr_harness \
  >> "$result/emitted_symbols.txt"

find "$result" -type f ! -name SHA256SUMS \
  -print0 | sort -z | xargs -0 sha256sum > "$result/SHA256SUMS"
