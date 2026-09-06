#!/usr/bin/env bash
set -u

repo=$(cd "$(dirname "$0")/../.." && pwd)
result_root="$repo/issues/polybench_section42"
util="$repo/tools/cgeist/Test/polybench/utilities"
build=/tmp/polygeist-section42-build
llvm=/home/arjaiswal/Polygeist/llvm-project/build/bin
export PATH="$build/bin:$llvm:$PATH"
export MLIR_OPT="$llvm/mlir-opt"
export MLIR_TRANSLATE="$llvm/mlir-translate"
export CLANG="$llvm/clang"
export POLYGEIST_CPU_BLAS=1
export POLYGEIST_CPU_BLAS_LIBS=-lopenblas

summary="$result_root/logs/cpu_library_sweep_summary.csv"
printf 'kernel,status,build_rc,run_rc,compare_rc,reference_sha256,candidate_sha256\n' > "$summary"

# These are the rows whose matches include at least one CBLAS operation.
# 3mm is cuTensorNet-only; covariance and deriche are memset-only and are not
# represented as OpenBLAS/CBLAS configurations.
for kernel in 2mm atax bicg doitgen gemm gemver gesummv gramschmidt mvt; do
  source_rel=$(awk -F, -v k="$kernel" '$1==k {print $3}' "$result_root/manifest.csv")
  source="$repo/$source_rel"
  function="kernel_${kernel//-/_}"
  matched="$result_root/ir/$kernel/matched.mlir"
  log_dir="$result_root/logs/$kernel"
  export_dir="$result_root/ir/$kernel/cpu_library"
  out="/tmp/polybench-section42-${kernel}-openblas-correctness"
  mkdir -p "$log_dir" "$export_dir"
  export POLYGEIST_EXPORT_OBJECT_DIR="$export_dir"
  "$repo/scripts/correctness/polygeist_build.sh" \
    --target=host --function="$function" --semantic-mlir="$matched" \
    -o "$out" "$source" -O3 -I"$util" -I"$(dirname "$source")" \
    -Dstatic= \
    -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_USE_C99_PROTO \
    -DPOLYBENCH_DUMP_ARRAYS > "$log_dir/cpu_library_build.log" 2>&1
  build_rc=$?
  if [[ $build_rc -eq 0 ]]; then
    nm "$out" | grep " $function$" > "$log_dir/cpu_library_symbol_audit.log" || true
    grep -q " T $function$" "$log_dir/cpu_library_symbol_audit.log" || build_rc=90
  fi
  run_rc=not_run
  compare_rc=not_run
  ref="$result_root/ir/$kernel/native_reference.out"
  candidate="$log_dir/cpu_library_output.txt"
  if [[ $build_rc -eq 0 ]]; then
    # The residual sweep's native output is copied once so every accepted
    # backend is compared to exactly the same canonical run.
    source_ref="/tmp/polybench-section42-residual/e2e_${kernel}_debuf/ref.out"
    if [[ -f "$source_ref" && ! -f "$ref" ]]; then cp "$source_ref" "$ref"; fi
    timeout 900s "$out" > "$log_dir/cpu_library_stdout.txt" 2> "$candidate"
    run_rc=$?
    if [[ $run_rc -eq 0 && -f "$ref" ]]; then
      "$repo/scripts/correctness/compare_polybench_dumps.py" \
        "$ref" "$candidate" --rtol 5e-4 --atol 1.1e-2 \
        > "$log_dir/cpu_library_correctness.log" 2>&1
      compare_rc=$?
    fi
  fi
  status=fail
  [[ $build_rc -eq 0 && $run_rc -eq 0 && $compare_rc -eq 0 ]] && status=pass
  ref_hash=unavailable
  candidate_hash=unavailable
  [[ -f "$ref" ]] && ref_hash=$(sha256sum "$ref" | awk '{print $1}')
  [[ -f "$candidate" ]] && candidate_hash=$(sha256sum "$candidate" | awk '{print $1}')
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$kernel" "$status" "$build_rc" \
    "$run_rc" "$compare_rc" "$ref_hash" "$candidate_hash" | tee -a "$summary"
done
