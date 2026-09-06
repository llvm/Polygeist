#!/usr/bin/env bash
set -u

repo=$(cd "$(dirname "$0")/../.." && pwd)
result_root="$repo/issues/polybench_section42"
util="$repo/tools/cgeist/Test/polybench/utilities"
clang=/home/arjaiswal/Polygeist/llvm-project/build/bin/clang
cpu=${POLYBENCH_TIMING_CPU:-21}
root=/tmp/polybench-section42-native-timing-final
mkdir -p "$root"
summary="$result_root/logs/native_timing_final_summary.csv"
printf 'kernel,configuration,status,build_rc,samples,cpu,threads\n' > "$summary"

while IFS=, read -r kernel category source_rel dataset datatype hash rest; do
  [[ "$kernel" == kernel ]] && continue
  source="$repo/$source_rel"
  out="$root/$kernel"
  log_dir="$result_root/logs/$kernel"
  mkdir -p "$out" "$log_dir"
  # Keep the timed kernel as a separately optimized function.  This prevents
  # whole-program dead-code elimination while retaining -O3 inside the kernel.
  "$clang" -O3 -fno-inline -fno-inline-functions \
    -I"$util" -I"$(dirname "$source")" -DLARGE_DATASET \
    -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_USE_C99_PROTO \
    -DPOLYBENCH_TIME \
    "$source" "$util/polybench.c" -lm -o "$out/native" \
    > "$log_dir/native_timing_final_build.log" 2>&1
  build_rc=$?
  status=fail
  samples=0
  : > "$log_dir/native_timing_final_raw.log"
  if [[ $build_rc -eq 0 ]]; then
    { nm -S "$out/native" | grep -E "kernel_${kernel//-/_}| main$" || true
      objdump -d --disassemble=main "$out/native" | \
        grep "kernel_${kernel//-/_}" || true
    } > "$log_dir/native_timing_final_symbol_audit.log"
    if ! grep -q "call.*<kernel_${kernel//-/_}>" "$log_dir/native_timing_final_symbol_audit.log"; then
      printf 'ERROR: no retained kernel symbol/call\n' >> "$log_dir/native_timing_final_symbol_audit.log"
      build_rc=91
    fi
  fi
  if [[ $build_rc -eq 0 ]]; then
    taskset -c "$cpu" timeout 900s "$out/native" >/dev/null 2>&1
    warmup_rc=$?
    if [[ $warmup_rc -eq 0 ]]; then
      status=pass
      for sample in 1 2 3 4 5; do
        value=$(taskset -c "$cpu" timeout 900s "$out/native" 2>/dev/null)
        rc=$?
        printf '%s,native_clang18_noinline,%d,%d,%s\n' \
          "$kernel" "$sample" "$rc" "$value" | \
          tee -a "$log_dir/native_timing_final_raw.log"
        if [[ $rc -ne 0 || ! "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
          status=fail
          break
        fi
        samples=$sample
      done
    fi
  fi
  printf '%s,native_clang18_noinline,%s,%d,%d,%s,1\n' \
    "$kernel" "$status" "$build_rc" "$samples" "$cpu" | tee -a "$summary"
done < "$result_root/manifest.csv"
