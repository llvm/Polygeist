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

summary="$result_root/logs/raised_gpu_sweep_summary.csv"
if [[ $# -gt 0 ]]; then
  summary="$result_root/logs/raised_gpu_sweep_targeted_summary.csv"
fi
printf 'kernel,status,build_rc,deploy_rc,run_rc,compare_rc,reference_sha256,candidate_sha256\n' > "$summary"

default_kernels=(2mm 3mm atax bicg covariance deriche doitgen gemm gemver gesummv gramschmidt mvt syr2k syrk trisolv symm trmm)
kernels=("${default_kernels[@]}")
if [[ $# -gt 0 ]]; then kernels=("$@"); fi
for kernel in "${kernels[@]}"; do
  source_rel=$(awk -F, -v k="$kernel" '$1==k {print $3}' "$result_root/manifest.csv")
  source="$repo/$source_rel"
  function="kernel_${kernel//-/_}"
  matched="$result_root/ir/$kernel/matched.mlir"
  log_dir="$result_root/logs/$kernel"
  export_dir="$result_root/ir/$kernel/raised_gpu"
  local_binary="/tmp/polybench-section42-${kernel}-jetson-correctness"
  remote_binary="/tmp/polybench-section42-${kernel}-jetson-correctness"
  mkdir -p "$log_dir" "$export_dir"
  export POLYGEIST_EXPORT_OBJECT_DIR="$export_dir"
  unset POLYGEIST_CUTENSORNET_ROOT POLYGEIST_MINIMAL_CUTENSORNET_RUNTIME
  export POLYGEIST_MINIMAL_CUDA_RUNTIME=1
  if [[ "$kernel" == 3mm ]]; then
    unset POLYGEIST_MINIMAL_CUDA_RUNTIME
    export POLYGEIST_CUTENSORNET_ROOT=/tmp/polygeist_cutensornet_aarch64/unified
    export POLYGEIST_MINIMAL_CUTENSORNET_RUNTIME=1
  fi
  "$repo/scripts/correctness/polygeist_build.sh" \
    --target=jetson --function="$function" --semantic-mlir="$matched" \
    -o "$local_binary" "$source" -O3 -I"$util" -I"$(dirname "$source")" \
    '-Dstatic=__attribute__((noipa))' \
    -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_USE_C99_PROTO \
    -DPOLYBENCH_DUMP_ARRAYS > "$log_dir/raised_gpu_build.log" 2>&1
  build_rc=$?
  if [[ $build_rc -eq 0 ]]; then
    aarch64-linux-gnu-nm "$local_binary" | grep " $function$" > "$log_dir/raised_gpu_symbol_audit.log" || true
    grep -q " T $function$" "$log_dir/raised_gpu_symbol_audit.log" || build_rc=90
  fi
  deploy_rc=not_run
  run_rc=not_run
  compare_rc=not_run
  ref="$result_root/ir/$kernel/native_reference.out"
  candidate="$log_dir/raised_gpu_output.txt"
  if [[ $build_rc -eq 0 ]]; then
    source_ref="/tmp/polybench-section42-residual/e2e_${kernel}_debuf/ref.out"
    if [[ -f "$source_ref" && ! -f "$ref" ]]; then cp "$source_ref" "$ref"; fi
    scp -o BatchMode=yes "$local_binary" pva-general:"${remote_binary}.stage" \
      > "$log_dir/raised_gpu_deploy.log" 2>&1 &&
      ssh pva-general "scp -o BatchMode=yes '${remote_binary}.stage' nvidia@192.168.57.1:'$remote_binary'" \
      >> "$log_dir/raised_gpu_deploy.log" 2>&1
    deploy_rc=$?
    if [[ $deploy_rc -eq 0 ]]; then
      ssh pva-general "ssh nvidia@192.168.57.1 'export LD_LIBRARY_PATH=/home/nvidia/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu; timeout 900s $remote_binary'" \
        > "$log_dir/raised_gpu_stdout.txt" 2> "$candidate"
      run_rc=$?
      if [[ $run_rc -eq 0 && -f "$ref" ]]; then
        "$repo/scripts/correctness/compare_polybench_dumps.py" \
          "$ref" "$candidate" --rtol 5e-4 --atol 1.1e-2 \
          > "$log_dir/raised_gpu_correctness.log" 2>&1
        compare_rc=$?
      fi
    fi
  fi
  status=fail
  [[ $build_rc -eq 0 && $deploy_rc -eq 0 && $run_rc -eq 0 && $compare_rc -eq 0 ]] && status=pass
  ref_hash=unavailable
  candidate_hash=unavailable
  [[ -f "$ref" ]] && ref_hash=$(sha256sum "$ref" | awk '{print $1}')
  [[ -f "$candidate" ]] && candidate_hash=$(sha256sum "$candidate" | awk '{print $1}')
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$kernel" "$status" "$build_rc" \
    "$deploy_rc" "$run_rc" "$compare_rc" "$ref_hash" "$candidate_hash" | tee -a "$summary"
done
