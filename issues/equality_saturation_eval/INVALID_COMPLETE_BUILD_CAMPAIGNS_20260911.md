# Invalid complete-build campaigns from 2026-09-11

Do not publish timing or build-coverage results from these directories:

- `polybench_compile_cost_correct_20260910_r3`
- `llama_compile_cost_correct_20260910_r3`
- `mfem_compile_cost_correct_20260910_r3`
- `aten_graph115_compile_cost_correct_20260911_r3`

The generic runners forced
`POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE=1` instead of using each suite's
validated production build profile.  This changes both successful compilation
paths and failure outcomes, so the successful timing samples are invalid as
well.  The ATen campaign was stopped after 257 persisted rows.  Raw artifacts
are retained solely for diagnosis and must not be used in paper tables.

Additional diagnosed issues are two generic wrapper-generator defects, missing
rank legality for `memset_zero_1D`, and an inconsistent FP64 strided-copy
contract.  Corrected campaigns must rebuild all inputs, not only failures.
