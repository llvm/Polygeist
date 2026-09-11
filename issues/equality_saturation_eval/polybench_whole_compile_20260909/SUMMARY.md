# PolyBench whole-compilation cost study

## Scope

Thirty canonical PolyBench/C 4.2 kernels were compiled from C source through a linked AArch64 executable. Each kernel used five fresh Egglog builds and five fresh syntactic-matcher builds, for 300 sequential attempts. Both arms disabled the handwritten semantic fallback. The input configuration was LARGE/FP64, each build had a 900-second outer timeout, and the production matcher used its eight-iteration and 32-AST-node proof bounds.

The total wall interval includes cgeist, affine-to-Linalg raising, submap lowering, debufferization, matching, library-definition injection, ABI lowering, MLIR-to-LLVM lowering, LLVM translation, AArch64 object compilation, runtime/harness compilation, and final linking. A generated non-computational harness retains the transformed kernel symbol solely so the linker produces the final artifact; executables were not run and these results make no runtime-correctness claim.

## Results

- Completed attempts: 300/300; no outer timeouts.
- Successful linked builds: 115 Egglog and 115 syntactic (23 kernels completed all five runs in both modes).
- Failed builds: 35 Egglog and 35 syntactic, covering the same seven kernels: adi, atax, bicg, doitgen, gemver, gesummv, mvt.
- Median successful whole-build wall time: 8.220 s Egglog versus 8.220 s syntactic.
- Median paired Egglog minus syntactic whole-build delta: -0.010 s; mean paired delta: +0.009 s across 115 pairs.
- Median successful peak RSS: 154.11 MiB Egglog versus 154.10 MiB syntactic; median paired delta +0.016 MiB.
- Median paired matcher-only overhead: +26.821 ms for Egglog.
- Selected launches across the five suites: 160 Egglog versus 145 syntactic, or 32 versus 29 per complete 30-kernel pass. Egglog found one additional 2mm launch and two additional gemver launches per pass; gemver subsequently failed ABI validation in both modes.
- Sum of measured attempt wall times, including failures: 33.13 minutes.

The seven failures are current compiler failures shared by both modes, not timeouts: malformed or mismatched launch/ABI IR after fresh source raising. They are reported directly and excluded from successful whole-build timing comparisons. Per-run logs retain the exact diagnostics.

## Caveats

The host load averages at campaign start were recorded in metadata.json and were high relative to an idle host. Five repetitions and alternating arm order reduce ordering bias, but these measurements should be labelled loaded-host results until repeated on an otherwise idle compilation machine. GNU time peak RSS is the maximum resident set reported for the build process tree, not the sum of simultaneously resident child processes. Unlike the isolated equality-saturation audit, the production matcher does not apply an explicit per-candidate wall timer; all measured maximum individual proofs were below 23 ms, so the paper's 10-second ceiling would not have censored any observed proof.

Raw data are in runs.csv, per-kernel aggregates in per_kernel.csv, and full provenance in metadata.json. All GNU-time records and matcher telemetry are retained, together with build logs for every failure. Successful verbose build logs, generated binaries, and link-only harness copies remain local rather than entering Git; their commands and executable hashes are recorded in runs.csv.
