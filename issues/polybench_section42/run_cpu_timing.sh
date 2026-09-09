#!/usr/bin/env bash
set -u

repo=$(cd "$(dirname "$0")/../.." && pwd)
result_root="$repo/issues/polybench_section42"
util="$repo/tools/cgeist/Test/polybench/utilities"
clang=/home/arjaiswal/Polygeist/llvm-project/build/bin/clang
llvm=/home/arjaiswal/Polygeist/llvm-project/build/bin
build=${POLYGEIST_BUILD:-/tmp/polygeist-section42-build}
cpu=${POLYBENCH_TIMING_CPU:-21}
skip_native=${POLYBENCH_SKIP_NATIVE:-0}
start_at=${POLYBENCH_START_AT:-}
library_only=${POLYBENCH_LIBRARY_ONLY:-0}
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
if [[ "$library_only" != 1 ]]; then
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

  # The former raised-residual path weakened the original kernel symbol so a
  # selected replacement won at link time. That is now forbidden and excluded.
  printf '%s,raised_residual_cpu,excluded,125,%s,%s,1\n' "$kernel" 0 "$cpu" | tee -a "$summary"
done < "$result_root/manifest.csv"
fi

# CPU-library application rows formerly used the same symbol-substitution
# mechanism through polygeist_build.sh. They are excluded until an untouched
# compiler transformation and application build are available.
