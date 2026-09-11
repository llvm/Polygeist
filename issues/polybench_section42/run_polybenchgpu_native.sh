#!/usr/bin/env bash
set -u

repo=$(cd "$(dirname "$0")/../.." && pwd)
result_root="$repo/issues/polybench_section42"
util="$repo/tools/cgeist/Test/polybench/utilities"
pbgpu_root="${POLYBENCHGPU_ROOT:-/home/arjaiswal/Polygeist/third_party/polybenchGpu}"
nvcc="${NVCC:-/usr/local/cuda-12.6/bin/nvcc}"
cuda_cross="${CUDA_CROSS:-/usr/local/cuda-12.6/targets/sbsa-linux}"
cc="${AARCH64_CC:-aarch64-linux-gnu-gcc}"
cxx="${AARCH64_CXX:-aarch64-linux-gnu-g++}"
expected_pbgpu_commit=5584aaa7d0be810ff5eb0b61c49fb64ecc81ba4c

actual_pbgpu_commit=$(git -C "$pbgpu_root" rev-parse HEAD 2>/dev/null || true)
if [[ "$actual_pbgpu_commit" != "$expected_pbgpu_commit" ]]; then
  echo "ERROR: expected PolyBenchGPU $expected_pbgpu_commit, got ${actual_pbgpu_commit:-unavailable}" >&2
  exit 1
fi

default_kernels=(gemm 2mm 3mm atax bicg correlation covariance doitgen
  fdtd-2d gemver gesummv gramschmidt mvt)
kernels=("${default_kernels[@]}")
if [[ $# -gt 0 ]]; then kernels=("$@"); fi
summary="$result_root/logs/polybenchgpu_native_summary.csv"
if [[ $# -gt 0 ]]; then
  summary="$result_root/logs/polybenchgpu_native_targeted_summary.csv"
fi
printf 'kernel,status,build_rc,run_rc,compare_rc,samples,source_commit,modified_source\n' > "$summary"

for kernel in "${kernels[@]}"; do
  source_rel=$(awk -F, -v k="$kernel" '$1==k {print $3}' "$result_root/manifest.csv")
  source="$repo/$source_rel"
  source_dir=$(dirname "$source")
  function="kernel_${kernel//-/_}"
  adapter="$result_root/polybenchgpu_adapters/$kernel.cu"
  log_dir="$result_root/logs/$kernel"
  work="/tmp/polybench-section42-native-gpu-$kernel"
  binary="$work/$kernel-correctness"
  timing_binary="$work/$kernel-timing"
  remote_binary="/tmp/polybench-section42-native-gpu-$kernel"
  mkdir -p "$log_dir" "$work"
  build_log="$log_dir/polybenchgpu_build.log"
  : > "$build_log"

  common_flags=(-O3 -fno-inline -fno-inline-functions
    -fsemantic-interposition -fgnu89-inline '-Dstatic=__attribute__((noipa))'
    -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_USE_C99_PROTO
    -I"$util" -I"$source_dir")
  echo "COMMAND: $cc ${common_flags[*]} -DPOLYBENCH_DUMP_ARRAYS -c $source" >> "$build_log"
  "$cc" "${common_flags[@]}" -DPOLYBENCH_DUMP_ARRAYS -c "$source" \
    -o "$work/harness_correctness_full.o" >> "$build_log" 2>&1
  build_rc=$?
  if [[ $build_rc -eq 0 ]]; then
    aarch64-linux-gnu-objcopy --weaken-symbol="$function" \
      "$work/harness_correctness_full.o" "$work/harness_correctness.o" \
      >> "$build_log" 2>&1 || build_rc=$?
  fi
  if [[ $build_rc -eq 0 ]]; then
    "$cc" -O2 -DPOLYBENCH_DUMP_ARRAYS -I"$util" -c "$util/polybench.c" \
      -o "$work/polybench_correctness.o" >> "$build_log" 2>&1 || build_rc=$?
  fi
  if [[ $build_rc -eq 0 ]]; then
    echo "COMMAND: $nvcc -O3 -std=c++17 -arch=sm_87 -ccbin $cxx -c $adapter" >> "$build_log"
    "$nvcc" -O3 -std=c++17 -arch=sm_87 -ccbin "$cxx" -I"$pbgpu_root" -c "$adapter" \
      -o "$work/adapter.o" >> "$build_log" 2>&1 || build_rc=$?
  fi
  if [[ $build_rc -eq 0 ]]; then
    "$cxx" -O3 "$work/harness_correctness.o" "$work/polybench_correctness.o" \
      "$work/adapter.o" -L"$cuda_cross/lib" -lcudart -lm \
      -Wl,-rpath,/home/nvidia/cuda-12.6/lib64 -o "$binary" \
      >> "$build_log" 2>&1 || build_rc=$?
  fi
  if [[ $build_rc -eq 0 ]]; then
    aarch64-linux-gnu-nm "$binary" | grep " T $function$" \
      > "$log_dir/polybenchgpu_symbol_audit.log" || build_rc=90
  fi

  run_rc=not_run
  compare_rc=not_run
  completed=0
  if [[ $build_rc -eq 0 ]]; then
    scp -o BatchMode=yes "$binary" pva-general:"${remote_binary}.stage" \
      > "$log_dir/polybenchgpu_deploy.log" 2>&1 &&
      ssh pva-general "scp -o BatchMode=yes '${remote_binary}.stage' nvidia@192.168.57.1:'$remote_binary'" \
      >> "$log_dir/polybenchgpu_deploy.log" 2>&1
    deploy_rc=$?
    if [[ $deploy_rc -eq 0 ]]; then
      ssh pva-general "ssh nvidia@192.168.57.1 'export LD_LIBRARY_PATH=/home/nvidia/cuda-12.6/lib64; timeout 900s $remote_binary'" \
        > "$log_dir/polybenchgpu_stdout.txt" 2> "$log_dir/polybenchgpu_output.txt"
      run_rc=$?
      if [[ $run_rc -eq 0 ]]; then
        "$repo/scripts/correctness/compare_polybench_dumps.py" \
          "$result_root/ir/$kernel/native_reference.out" \
          "$log_dir/polybenchgpu_output.txt" --rtol 5e-4 --atol 1.1e-2 \
          > "$log_dir/polybenchgpu_correctness.log" 2>&1
        compare_rc=$?
      fi
    fi
  fi

  if [[ $build_rc -eq 0 && $run_rc -eq 0 && $compare_rc -eq 0 ]]; then
    timing_log="$log_dir/polybenchgpu_timing_build.log"
    : > "$timing_log"
    "$cc" "${common_flags[@]}" -DPOLYBENCH_TIME -c "$source" \
      -o "$work/harness_timing_full.o" >> "$timing_log" 2>&1 || build_rc=$?
    if [[ $build_rc -eq 0 ]]; then
      aarch64-linux-gnu-objcopy --weaken-symbol="$function" \
        "$work/harness_timing_full.o" "$work/harness_timing.o" \
        >> "$timing_log" 2>&1 || build_rc=$?
      "$cc" -O2 -DPOLYBENCH_TIME -I"$util" -c "$util/polybench.c" \
        -o "$work/polybench_timing.o" >> "$timing_log" 2>&1 || build_rc=$?
    fi
    if [[ $build_rc -eq 0 ]]; then
      "$cxx" -O3 "$work/harness_timing.o" "$work/polybench_timing.o" \
        "$work/adapter.o" -L"$cuda_cross/lib" -lcudart -lm \
        -Wl,-rpath,/home/nvidia/cuda-12.6/lib64 -o "$timing_binary" \
        >> "$timing_log" 2>&1 || build_rc=$?
    fi
    if [[ $build_rc -eq 0 ]]; then
      scp -o BatchMode=yes "$timing_binary" pva-general:"${remote_binary}-timing.stage" \
        > "$log_dir/polybenchgpu_timing_deploy.log" 2>&1 &&
        ssh pva-general "scp -o BatchMode=yes '${remote_binary}-timing.stage' nvidia@192.168.57.1:'${remote_binary}-timing'" \
        >> "$log_dir/polybenchgpu_timing_deploy.log" 2>&1
      : > "$log_dir/polybenchgpu_timing_raw.log"
      for sample in 0 1 2 3 4 5; do
        stdout="$log_dir/polybenchgpu_timing_${sample}.stdout"
        stderr="$log_dir/polybenchgpu_timing_${sample}.stderr"
        ssh pva-general "ssh nvidia@192.168.57.1 'export LD_LIBRARY_PATH=/home/nvidia/cuda-12.6/lib64; timeout 900s ${remote_binary}-timing'" \
          > "$stdout" 2> "$stderr"
        rc=$?
        e2e_s=$(awk '/^[0-9]+([.][0-9]+)?$/ {value=$1} END {print value}' "$stdout")
        device_ms=$(sed -n 's/.*POLYBENCH_NATIVE_GPU_TIMING device_ms=\([0-9.]*\).*/\1/p' "$stderr" | tail -1)
        if [[ -z "$device_ms" ]]; then
          device_s=$(awk '/^[0-9]+([.][0-9]+)?$/ {print; exit}' "$stdout")
          if [[ -z "$device_s" ]]; then
            device_s=$(sed -n 's/^GPU Runtime: \([0-9.]*\)s$/\1/p' "$stdout" | head -1)
          fi
          [[ -n "$device_s" ]] && device_ms=$(awk -v t="$device_s" 'BEGIN {printf "%.9f", 1000*t}')
        fi
        printf '%s,%s,%s,%s,%s\n' "$kernel" "$sample" "$rc" \
          "${e2e_s:-unavailable}" "${device_ms:-unavailable}" \
          >> "$log_dir/polybenchgpu_timing_raw.log"
        [[ $sample -eq 0 ]] && continue
        [[ $rc -eq 0 && -n "$e2e_s" && -n "$device_ms" ]] || break
        completed=$sample
      done
    fi
  fi
  status=fail
  [[ $build_rc -eq 0 && $run_rc -eq 0 && $compare_rc -eq 0 && $completed -eq 5 ]] && status=pass
  printf '%s,%s,%s,%s,%s,%s,%s,true\n' "$kernel" "$status" "$build_rc" \
    "$run_rc" "$compare_rc" "$completed" "$actual_pbgpu_commit" | tee -a "$summary"
done
