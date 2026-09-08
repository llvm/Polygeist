#!/usr/bin/env bash
set -u

repo=$(cd "$(dirname "$0")/../.." && pwd)
result_root="$repo/issues/polybench_section42"
util="$repo/tools/cgeist/Test/polybench/utilities"
build=${POLYGEIST_BUILD:-/tmp/polygeist-section42-build}
llvm=/home/arjaiswal/Polygeist/llvm-project/build/bin
export PATH="$build/bin:$llvm:$PATH"
export MLIR_OPT="$llvm/mlir-opt"
export MLIR_TRANSLATE="$llvm/mlir-translate"
export CLANG="$llvm/clang"
export POLYGEIST_CUDA_TIMING_WRAPPER=1

summary="$result_root/logs/raised_gpu_timing_summary.csv"
printf 'kernel,status,build_rc,deploy_rc,samples\n' > "$summary"

default_kernels=(2mm 3mm atax bicg doitgen gemm gemver gesummv mvt syr2k syrk trisolv symm trmm)
kernels=("${default_kernels[@]}")
if [[ $# -gt 0 ]]; then kernels=("$@"); fi
for kernel in "${kernels[@]}"; do
  source_rel=$(awk -F, -v k="$kernel" '$1==k {print $3}' "$result_root/manifest.csv")
  source="$repo/$source_rel"
  function="kernel_${kernel//-/_}"
  matched="$result_root/ir/$kernel/matched.mlir"
  log_dir="$result_root/logs/$kernel"
  local_binary="/tmp/polybench-section42-${kernel}-jetson-timed"
  remote_binary="/tmp/polybench-section42-${kernel}-jetson-timed"
  raw="$log_dir/raised_gpu_timing_raw.log"
  : > "$raw"
  unset POLYGEIST_CUTENSORNET_ROOT POLYGEIST_MINIMAL_CUTENSORNET_RUNTIME
  export POLYGEIST_MINIMAL_CUDA_RUNTIME=1
  if grep -q 'kernel.launch @cutensornet' "$matched"; then
    unset POLYGEIST_MINIMAL_CUDA_RUNTIME
    export POLYGEIST_CUTENSORNET_ROOT=/tmp/polygeist_cutensornet_aarch64/unified
    export POLYGEIST_MINIMAL_CUTENSORNET_RUNTIME=1
  fi
  "$repo/scripts/correctness/polygeist_build.sh" \
    --target=jetson --function="$function" --semantic-mlir="$matched" \
    -o "$local_binary" "$source" -O3 -I"$util" -I"$(dirname "$source")" \
    '-Dstatic=__attribute__((noipa))' \
    -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_USE_C99_PROTO \
    -DPOLYBENCH_TIME > "$log_dir/raised_gpu_timing_build.log" 2>&1
  build_rc=$?
  if [[ $build_rc -eq 0 ]]; then
    aarch64-linux-gnu-nm "$local_binary" | grep " $function$" > "$log_dir/raised_gpu_timing_symbol_audit.log" || true
    grep -q " T $function$" "$log_dir/raised_gpu_timing_symbol_audit.log" || build_rc=90
  fi
  deploy_rc=not_run
  completed=0
  if [[ $build_rc -eq 0 ]]; then
    scp -o BatchMode=yes "$local_binary" pva-general:"${remote_binary}.stage" \
      > "$log_dir/raised_gpu_timing_deploy.log" 2>&1 &&
      ssh pva-general "scp -o BatchMode=yes '${remote_binary}.stage' nvidia@192.168.57.1:'$remote_binary'" \
      >> "$log_dir/raised_gpu_timing_deploy.log" 2>&1
    deploy_rc=$?
    if [[ $deploy_rc -eq 0 ]]; then
      # One complete process warmup is excluded from the five recorded runs.
      ssh pva-general "ssh nvidia@192.168.57.1 'export LD_LIBRARY_PATH=/home/nvidia/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu; timeout 900s $remote_binary'" \
        > "$log_dir/raised_gpu_timing_warmup.stdout" \
        2> "$log_dir/raised_gpu_timing_warmup.stderr"
      warmup_rc=$?
      if [[ $warmup_rc -ne 0 ]]; then
        printf '%s,warmup,%d,unavailable,unavailable\n' \
          "$kernel" "$warmup_rc" | tee -a "$raw"
      else
        for sample in 1 2 3 4 5; do
        stdout="$log_dir/raised_gpu_timing_${sample}.stdout"
        stderr="$log_dir/raised_gpu_timing_${sample}.stderr"
        ssh pva-general "ssh nvidia@192.168.57.1 'export LD_LIBRARY_PATH=/home/nvidia/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu; timeout 900s $remote_binary'" \
          > "$stdout" 2> "$stderr"
        rc=$?
        e2e_s=$(awk '/^[0-9]+([.][0-9]+)?$/ {value=$1} END {print value}' "$stdout")
        device_ms=$(sed -n 's/.*POLYGEIST_DEVICE_TIMING.*device_ms=\([0-9.]*\).*/\1/p' "$stderr" | tail -1)
        printf '%s,%d,%d,%s,%s\n' "$kernel" "$sample" "$rc" "${e2e_s:-unavailable}" "${device_ms:-unavailable}" | tee -a "$raw"
        [[ $rc -eq 0 && -n "$e2e_s" && -n "$device_ms" ]] || break
        completed=$sample
        done
      fi
    fi
  fi
  status=fail
  [[ $build_rc -eq 0 && $deploy_rc -eq 0 && $completed -eq 5 ]] && status=pass
  printf '%s,%s,%s,%s,%s\n' "$kernel" "$status" "$build_rc" "$deploy_rc" "$completed" | tee -a "$summary"
done
