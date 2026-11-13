ATen reproduction checkpoint
===========================

This checkpoint records the ATen fixes and available evidence. It is not a
completed artifact submission. Matching uses the existing phase-2 pipeline,
including structural recognition; these results do not establish an end-to-end
lib_v3 or equality-saturation ablation result.

Validated performance suite
---------------------------
The final fresh build and GPU run cover all 115 Figure 5 cases. All 115 raised
outputs pass; the retained native comparisons also pass all 115 cases.
retained-timings.json preserves the five samples, correctness outcomes,
executable hashes, and source-summary paths for both sides. Native timings are
aggregated from the earlier validated runs; they were not all rerun alongside
the final raised sweep. result-comparison.csv records attribution per case,
including the 26 ATen compositions for extracted stages.

The current within-2x / over-2x-faster / over-2x-slower split is 66 / 17 / 32;
the reported paper split is 66 / 15 / 34. These are not identical results.
Two historical inputs cannot be compared directly: conv2d_columns used an
invalid scaled kernel width, and sparse_matmul used an empty CSR matrix.
Their repaired inputs are explicitly marked in result-comparison.csv.

Runs used one CPU core, one library thread, MAXN, fixed CPU/GPU clocks, five
warmups and five synchronized measurements. The minimum is the headline.
The board fan remained automatic, so fixed-fan methodology is still unmet.
Full local build products and logs remain in this directory but are not all
included in this source checkpoint.

Coverage and pending work
-------------------------
coverage-checkpoint.json distinguishes the full 598-case sweep (260 complete
ABI mappings) from three later targeted checks (conv3d, unit-batch transpose2d,
and softmax). Combined, they establish 263 complete mappings, including 28/52
native-library cases, but a fresh consolidated sweep is still required.
There are 11 missing historical cases and three additional cases; the paper
reported 271. Complete ABI mappings do not by themselves establish GPU safety
or runtime correctness. The three latest coverage variants still need GPU
validation, followed by the latest Llama/PolyBench regression checks.

The final 115-case build predates those last three coverage edits. Its exact
source hashes are in final-build-provenance.json. Do not interpret that run as
runtime validation of every edit in this checkpoint.

Checks retained
---------------
The previous core suite had 274 passes, two expected failures, six known
failures, and 14 unresolved tests. Before committing, all 33 added matcher/pass
tests and the added cgeist logical-result test passed. Their logs are included.
GPU boundary checks passed 512 prefix-bound cases, 256 joint-maxabs cases,
96 product cases, and 32 arg-reduction cases (cub-prefix-regression.json).

Entry points
------------
build_final.py builds the pinned 115 cases from paper_targets.csv through
scripts/correctness/aten_pointwise_graph_silicon.py. Use
scripts/correctness/reproduce_aten_coverage.py for a fresh coverage sweep.
scripts/correctness/prepare_aten_native_sysroot.py obtains the pinned public
ATen cross-build dependencies; build_aten_native_fixture_extensions.py builds
the additional native comparisons. Hardware access stays in private local
configuration and is not part of this checkpoint.
