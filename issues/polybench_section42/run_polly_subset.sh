#!/usr/bin/env bash
set -u

repo=$(cd "$(dirname "$0")/../.." && pwd)
result_root="$repo/issues/polybench_section42"
util="$repo/tools/cgeist/Test/polybench/utilities"
clang=/usr/bin/clang-14
cpu=${POLYBENCH_TIMING_CPU:-21}

for kernel in gemm syr2k 2mm 3mm; do
  source=$(awk -F, -v k="$kernel" '$1==k {print $3}' "$result_root/manifest.csv")
  source="$repo/$source"
  source_dir=$(dirname "$source")
  out=/tmp/polybench-section42-polly/$kernel
  log_dir="$result_root/logs/$kernel"
  mkdir -p "$out" "$log_dir"
  common=(-O3 -fno-inline -fno-inline-functions -I"$util" -I"$source_dir"
          -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_USE_C99_PROTO)
  polly=(-mllvm -polly -mllvm -polly-process-unprofitable)

  "$clang" "${common[@]}" -DPOLYBENCH_DUMP_ARRAYS \
    "$source" "$util/polybench.c" -lm -o "$out/reference"
  "$clang" "${common[@]}" "${polly[@]}" -DPOLYBENCH_DUMP_ARRAYS \
    "$source" "$util/polybench.c" -lm -o "$out/polly_correctness"
  taskset -c "$cpu" timeout 900s "$out/reference" 2> "$out/reference.out"
  ref_rc=$?
  taskset -c "$cpu" timeout 900s "$out/polly_correctness" 2> "$out/polly.out"
  polly_rc=$?
  if [[ $ref_rc -eq 0 && $polly_rc -eq 0 ]] && \
     cmp -s "$out/reference.out" "$out/polly.out"; then
    correctness=pass
  else
    correctness=fail
  fi
  printf 'correctness=%s reference_rc=%d polly_rc=%d\n' \
    "$correctness" "$ref_rc" "$polly_rc" | tee "$log_dir/polly_correctness.log"
  [[ "$correctness" == pass ]] || continue

  "$clang" "${common[@]}" -DPOLYBENCH_TIME \
    "$source" "$util/polybench.c" -lm -o "$out/native_timed"
  "$clang" "${common[@]}" "${polly[@]}" -DPOLYBENCH_TIME \
    "$source" "$util/polybench.c" -lm -o "$out/polly_timed"
  : > "$log_dir/polly_timing_raw.log"
  for configuration in native_clang14 polly14; do
    binary="$out/native_timed"
    [[ "$configuration" == polly14 ]] && binary="$out/polly_timed"
    taskset -c "$cpu" timeout 900s "$binary" >/dev/null 2>&1
    for sample in 1 2 3 4 5; do
      value=$(taskset -c "$cpu" timeout 900s "$binary" 2>/dev/null)
      rc=$?
      printf '%s,%d,%d,%s\n' "$configuration" "$sample" "$rc" "$value" | \
        tee -a "$log_dir/polly_timing_raw.log"
      [[ $rc -eq 0 ]] || break
    done
  done
done
