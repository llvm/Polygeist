* PolyBench fresh compilation campaign (3 repetitions)

Scope: 30 canonical PolyBench/C 4.2 kernels, LARGE/FP64, compiled from C to a
linked AArch64 executable.  Each kernel was compiled three times with Egglog
enabled and three times with the syntactic ablation, for 180 sequential fresh
builds.  The semantic fallback was disabled.  Every invocation used a new
temporary build directory and deleted its affine, Linalg, matched, LLVM, and
object intermediates on exit.  The final executable, raw logs, timing records,
telemetry, commands, and hashes were retained but never reused.

The total interval includes cgeist, affine-to-Linalg raising, debufferization,
matching, library-definition injection, library ABI lowering, MLIR/LLVM
lowering, AArch64 kernel/runtime/harness compilation, and final linking.  The
runtime shim was rebuilt on every invocation.  Executables were not run, so
this campaign measures compilation and makes no new runtime-correctness claim.

Results:

- Scheduled and completed: 180/180; no timeouts.
- Successful linked builds: 168 (84 Egglog and 84 syntactic).
- Complete kernels: 28/30 in both modes and all three repetitions.
- Failed attempts: 12, covering `adi` and `doitgen` in both modes and all three
  repetitions.
- Median of the 28 per-kernel three-run medians: 8.960 s Egglog and 8.955 s
  syntactic.
- Mean of the 28 per-kernel medians: 9.012 s Egglog and 9.029 s syntactic.
- Range of per-kernel medians: 8.55--9.46 s Egglog and 8.66--9.51 s
  syntactic.
- Median per-kernel Egglog-minus-syntactic delta: 0.000 s; mean delta:
  -0.018 s.
- Median paired matcher-only Egglog overhead: 13.754 ms across 84 successful
  kernel/repetition pairs.
- Median peak RSS: 153.260 MiB Egglog and 153.291 MiB syntactic.
- Median successful Egglog stage times: frontend 0.104 s, raising 0.024 s,
  matching process 0.950 s, definition injection 0.088 s, library lowering
  0.176 s, downstream lowering 0.099 s, ABI-wrapper preparation 0.042 s,
  native/runtime/harness compilation 6.632 s, and linking 0.696 s.
- Sum of all attempt wall times: 26.887 minutes. Campaign elapsed wall time:
  26.976 minutes.

Failures:

- `adi`: library ABI lowering rejects a subview with `expected 1 offset values,
  got 2`.
- `doitgen`: cuTensorNet composition produces a launch type mismatch between
  `memref<?xf64>` and `memref<?xf64, strided<[1], offset: ?>>`.

Important validity caveat: host load averages at campaign start were
31.08/19.77/17.22 on a 24-logical-CPU shared machine.  Other users' workloads
were visible during the campaign.  The alternating arm order and paired
comparison make the Egglog-vs-syntactic delta useful, but absolute compilation
times must be labelled loaded-host measurements and should not replace an
idle-host publication result.  The production matcher also uses iteration and
AST-size bounds instead of an explicit ten-second per-candidate timer; the
observed maximum proof time was 16.47 ms, so a ten-second limit would not have
censored an observed proof.

Revision: `7d69dcc9928448e775d9e671992aa92e25386754`.  Exact binary, runner,
build-driver, manifest, and source hashes are in `metadata.json` and
`runs.csv`.  The primary worktree was dirty; the artifacts therefore identify
the actual consumed compiler binaries and sources by SHA-256 rather than
claiming that the Git commit alone fully describes the run.
