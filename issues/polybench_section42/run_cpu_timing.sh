#!/usr/bin/env bash
set -u

repo=$(cd "$(dirname "$0")/../.." && pwd)
result_root="$repo/issues/polybench_section42"
util="$repo/tools/cgeist/Test/polybench/utilities"
clang=/home/arjaiswal/Polygeist/llvm-project/build/bin/clang
llvm=/home/arjaiswal/Polygeist/llvm-project/build/bin
build=/tmp/polygeist-section42-build
cpu=${POLYBENCH_TIMING_CPU:-21}
skip_native=${POLYBENCH_SKIP_NATIVE:-0}
start_at=${POLYBENCH_START_AT:-}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PATH="$build/bin:$llvm:$PATH"
export MLIR_OPT="$llvm/mlir-opt"
export MLIR_TRANSLATE="$llvm/mlir-translate"
export CLANG="$clang"

root=/tmp/polybench-section42-cpu-timing
mkdir -p "$root"
summary="$result_root/logs/cpu_timing_build_summary.csv"
if [[ -z "$start_at" ]]; then
  printf 'kernel,configuration,status,build_rc,samples,cpu,threads\n' > "$summary"
else
  touch "$summary"
fi

run_samples() {
  local kernel=$1 configuration=$2 binary=$3 log=$4
  : > "$log"
  local completed=0
  # One unrecorded warmup in the same pinned execution environment.
  taskset -c "$cpu" timeout 900s "$binary" >/dev/null 2>&1 || return 1
  for sample in 1 2 3 4 5; do
    value=$(taskset -c "$cpu" timeout 900s "$binary" 2>/dev/null)
    rc=$?
    printf '%s,%s,%d,%d,%s\n' "$kernel" "$configuration" "$sample" "$rc" "$value" | tee -a "$log"
    [[ $rc -eq 0 && "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || return 1
    completed=$sample
  done
  [[ $completed -eq 5 ]]
}

started=0
while IFS=, read -r kernel category source_rel dataset datatype hash native raise matcher residual rest; do
  [[ "$kernel" == kernel || "$residual" != pass ]] && continue
  if [[ -n "$start_at" && $started -eq 0 ]]; then
    [[ "$kernel" == "$start_at" ]] || continue
    started=1
  fi
  source="$repo/$source_rel"
  source_dir=$(dirname "$source")
  function="kernel_${kernel//-/_}"
  out="$root/$kernel"
  log_dir="$result_root/logs/$kernel"
  mkdir -p "$out" "$log_dir"
  common=(-I"$util" -I"$source_dir" -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE
          -DPOLYBENCH_USE_C99_PROTO -DPOLYBENCH_TIME)

  if [[ "$skip_native" != 1 ]]; then
    "$clang" -O3 "${common[@]}" "$source" "$util/polybench.c" -lm \
      -o "$out/native" > "$log_dir/native_timing_build.log" 2>&1
    native_build=$?
    native_status=fail
    if [[ $native_build -eq 0 ]] && run_samples "$kernel" native_clang18 "$out/native" "$log_dir/native_timing_raw.log"; then
      native_status=pass
    fi
    printf '%s,native_clang18,%s,%d,%s,%s,1\n' "$kernel" "$native_status" "$native_build" 5 "$cpu" | tee -a "$summary"
  fi

  cp "/tmp/polybench-section42-residual/e2e_${kernel}_debuf/wrapper.c" "$out/residual_wrapper.c"
  "$clang" -O3 -fno-inline -fno-inline-functions \
    -fsemantic-interposition -fgnu89-inline \
    "${common[@]}" -Dstatic= -c "$source" -o "$out/harness_full.o" &&
    objcopy --weaken-symbol="$function" "$out/harness_full.o" "$out/harness.o" &&
    "$clang" -O3 "${common[@]}" -c "$util/polybench.c" -o "$out/polybench.o" &&
    "$clang" -O3 -c "$out/residual_wrapper.c" -o "$out/residual_wrapper.o" &&
    "$clang" -O3 -c "$result_root/ir/$kernel/residual.ll" -o "$out/residual_kernel.o" &&
    "$clang" "$out/harness.o" "$out/residual_wrapper.o" "$out/residual_kernel.o" \
      "$out/polybench.o" -lm -L/home/arjaiswal/Polygeist/llvm-project/build/lib \
      -Wl,-rpath,/home/arjaiswal/Polygeist/llvm-project/build/lib \
      -lmlir_c_runner_utils -o "$out/residual" \
      > "$log_dir/residual_timing_build.log" 2>&1
  residual_build=$?
  residual_status=fail
  if [[ $residual_build -eq 0 ]] && nm "$out/residual" | grep -q " T $function$" && \
      run_samples "$kernel" raised_residual_cpu "$out/residual" "$log_dir/residual_timing_raw.log"; then
    residual_status=pass
  fi
  printf '%s,raised_residual_cpu,%s,%d,%s,%s,1\n' "$kernel" "$residual_status" "$residual_build" 5 "$cpu" | tee -a "$summary"
done < "$result_root/manifest.csv"

for kernel in 2mm atax bicg gemm gemver gesummv mvt; do
  source_rel=$(awk -F, -v k="$kernel" '$1==k {print $3}' "$result_root/manifest.csv")
  source="$repo/$source_rel"
  function="kernel_${kernel//-/_}"
  out="$root/$kernel/openblas"
  log_dir="$result_root/logs/$kernel"
  export POLYGEIST_CPU_BLAS=1
  export POLYGEIST_CPU_BLAS_LIBS=-lopenblas
  "$repo/scripts/correctness/polygeist_build.sh" --target=host \
    --function="$function" --semantic-mlir="$result_root/ir/$kernel/matched.mlir" \
    -o "$out" "$source" -O3 -I"$util" -I"$(dirname "$source")" -Dstatic= \
    -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_USE_C99_PROTO \
    -DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS \
    > "$log_dir/cpu_library_timing_build.log" 2>&1
  build_rc=$?
  status=fail
  if [[ $build_rc -eq 0 ]] && nm "$out" | grep -q " T $function$" && \
      run_samples "$kernel" openblas_cblas_1t "$out" "$log_dir/cpu_library_timing_raw.log"; then
    status=pass
  fi
  printf '%s,openblas_cblas_1t,%s,%d,%s,%s,1\n' "$kernel" "$status" "$build_rc" 5 "$cpu" | tee -a "$summary"
done
