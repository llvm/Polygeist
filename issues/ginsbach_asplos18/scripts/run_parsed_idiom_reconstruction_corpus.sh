#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
audit_dir="$(cd "${script_dir}/.." && pwd)"
clang_bin="${audit_dir}/build/candl-parsed-reconstruction/bin/clang"
run_root="${audit_dir}/runs/parsed-idl-reconstruction-corpus"
log_root="${audit_dir}/logs"
runner="${script_dir}/run_idl_matcher_corpus.py"

mkdir -p "${run_root}" "${log_root}"

run_shard() {
  local name="$1"
  shift
  local program_args=()
  for program in "$@"; do
    program_args+=(--program "${program}")
  done
  "${runner}" --clang "${clang_bin}" --out "${run_root}/${name}" \
    --timeout 300 "${program_args[@]}" \
    >"${log_root}/run_parsed_idiom_${name}.log" 2>&1
}

printf 'timestamp=%s\n' "$(date -Is)"
printf 'classification=experimental reconstruction; never count as original artifact\n'

run_shard shard1 BT CG DC EP FT & p1=$!
run_shard shard2 IS LU MG & p2=$!
run_shard shard3 SP UA & p3=$!
run_shard shard4 bfs cutcp histo lbm mri-gridding mri-q sad sgemm spmv stencil tpacf & p4=$!

status=0
for pid in "${p1}" "${p2}" "${p3}" "${p4}"; do
  wait "${pid}" || status=1
done

combined="${run_root}/translation_units.csv"
head -n 1 "${run_root}/shard1/translation_units.csv" > "${combined}"
for shard in shard1 shard2 shard3 shard4; do
  tail -n +2 "${run_root}/${shard}/translation_units.csv" >> "${combined}"
done

printf 'rows=%s\n' "$(($(wc -l < "${combined}") - 1))"
printf 'compile_or_timeout_status=%s\n' "${status}"
exit "${status}"
