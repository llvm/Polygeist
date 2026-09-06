#!/usr/bin/env bash
set -u

repo=$(cd "$(dirname "$0")/../.." && pwd)
result_root="$repo/issues/polybench_section42"
: > "$result_root/logs/matcher_sweep_summary.csv"

while IFS=, read -r kernel category source dataset datatype hash native raise matcher residual rest; do
  [[ "$kernel" == kernel ]] && continue
  log_dir="$result_root/logs/$kernel"
  ir_dir="$result_root/ir/$kernel"
  if [[ "$residual" != pass ]]; then
    printf '%s,unavailable,0,,residual_not_correct\n' "$kernel" | \
      tee -a "$result_root/logs/matcher_sweep_summary.csv"
    continue
  fi
  start=$(date -Is)
  timeout 300s /usr/bin/time -f 'wall_seconds=%e maxrss_kb=%M' \
    /usr/bin/python3 "$repo/scripts/correctness/kernel_match_rewrite.py" \
    "$ir_dir/raised_debufferized.mlir" > "$ir_dir/matched.mlir" \
    2> "$log_dir/matcher.log"
  rc=$?
  end=$(date -Is)
  launches=$(grep -cE '\bkernel\.launch ' "$ir_dir/matched.mlir" 2>/dev/null || true)
  symbols=$(sed -n 's/.*kernel\.launch @\([^ (]*\).*/\1/p' "$ir_dir/matched.mlir" | \
    sort -u | paste -sd+ -)
  printf '%s,%d,%s,%s,%s,%s\n' \
    "$kernel" "$rc" "${launches:-0}" "$symbols" "$start" "$end" | \
    tee -a "$result_root/logs/matcher_sweep_summary.csv"
done < "$result_root/manifest.csv"
