#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
audit_dir="$(cd "${script_dir}/.." && pwd)"
source_csv="${audit_dir}/published_idiom_manifest.csv"
ginsbach_csv="${audit_dir}/ginsbach_60_manifest.csv"
polygeist_csv="${audit_dir}/polygeist_same_inputs.csv"

# This is a bootstrap operation.  Once occurrence evidence has been audited,
# regenerating the skeletons would destroy that evidence.  Require an explicit
# force flag rather than overwriting reviewed manifests silently.
if [[ "${1:-}" != "--force" ]] &&
   { [[ -e "${ginsbach_csv}" ]] || [[ -e "${polygeist_csv}" ]]; }; then
  printf 'refusing to overwrite existing audited manifests; use --force only to recreate unresolved skeletons\n' >&2
  exit 2
fi

# Figure 16 identifies instances only to suite/program/category.  This script
# deliberately leaves source and IR locations unresolved instead of guessing.
awk -F, 'function q(s) { gsub(/"/, "\"\"", s); return "\"" s "\"" } BEGIN { OFS="," }
NR == 1 {
  print q("occurrence_id"),q("benchmark_suite"),q("program"),q("source_revision"),q("source_file_and_function"),q("llvm_ir_function_region"),q("idiom_category"),q("replacement_library_api"),q("evidence_source"),q("original_tool_match_status"),q("replacement_generation_status"),q("build_status"),q("correctness_status"),q("runtime_status"),q("blocker_notes")
  next
}
{
  api="not reported per occurrence"
  if ($4 == "scalar_reduction" || $4 == "histogram_reduction")
    api=api "; paper supports Lift"
  else if ($4 == "stencil")
    api=api "; paper supports Halide and Lift"
  else if ($4 == "matrix_operation")
    api=api "; paper supports cuBLAS, clBLAS, CLBlast, MKL, and Lift"
  else if ($4 == "sparse_matrix_operation")
    api=api "; paper supports cuSPARSE, clSPARSE, MKL, and custom libSPMV"

  revision=($2 == "snu-npb") ? "paper revision not reported; local mirror 4f2aa1b4d3127dbb3612c8aef24b24c69e83013c unverified" : "paper revision not reported; local mirror ccf3d3126f1754ca85528722f4ced5894ccb852b unverified"
  print q($1),q($2),q($3),q(revision),q("unresolved"),q("unresolved"),q($4),q(api),q("Ginsbach et al. ASPLOS 2018 Figure 16 and Table 1"),q("reported detected; not independently reproduced"),q("not evaluated"),q("not evaluated"),q("not evaluated"),q("paper runtime coverage only; occurrence runtime not reported"),q("Exact source/IR occurrence and original benchmark revision unavailable")
}' "${source_csv}" > "${ginsbach_csv}"

awk -F, 'function q(s) { gsub(/"/, "\"\"", s); return "\"" s "\"" } BEGIN { OFS="," }
NR == 1 {
  print q("occurrence_id"),q("benchmark_suite"),q("program"),q("source_revision"),q("source_file_and_function"),q("llvm_ir_function_region"),q("idiom_category"),q("evidence_source"),q("source_translation_status"),q("raised_loop_count"),q("linalg_generic_count"),q("semantic_candidate_count"),q("kernel_launch_count"),q("lowered_runtime_calls"),q("executable_generation_status"),q("correctness_status"),q("silicon_status"),q("performance_status"),q("failure_class"),q("blocker_notes")
  next
}
{
  revision=($2 == "snu-npb") ? "paper revision not reported; local mirror 4f2aa1b4d3127dbb3612c8aef24b24c69e83013c unverified" : "paper revision not reported; local mirror ccf3d3126f1754ca85528722f4ced5894ccb852b unverified"
  print q($1),q($2),q($3),q(revision),q("unresolved"),q("unresolved"),q($4),q("Ginsbach et al. ASPLOS 2018 Figure 16 and Table 1"),q("not evaluated: paper occurrence unmapped"),q("not attributable"),q("not attributable"),q("not attributable"),q("not attributable"),q("not attributable"),q("not evaluated"),q("not evaluated"),q("not evaluated"),q("not comparable"),q("unavailable source/artifact"),q("Polygeist result cannot be attributed until the paper occurrence is mapped to an exact source/IR region")
}' "${source_csv}" > "${polygeist_csv}"

test "$(($(wc -l < "${ginsbach_csv}") - 1))" -eq 60
test "$(($(wc -l < "${polygeist_csv}") - 1))" -eq 60

printf 'initialized %s and %s with 60 occurrence rows each\n' \
  "${ginsbach_csv}" "${polygeist_csv}"
