# MFEM matcher audit — 2026-09-07

## Scope

This audit covers the current MFEM-derived normalized fixtures, larger
application/operator extractions, semantic contraction matcher, textual
rewrite, checked-in result ledgers, correctness evidence, and performance
claims.  The audit began at revision
`e4464a64b00134b6279c1449dfe9b5de9d118c9f`; during the audit the branch
advanced to `cfb0e9756574a213d028e1c192864c088a17440d`.  That commit added
PolyBench routes but did not change the MFEM contraction recognizer.  The
workspace also contains uncommitted changes.  Live counts below therefore
describe the audited working tree and must not be attributed to either commit
alone until the campaign is regenerated from an isolated snapshot.

## Name-independence result

No MFEM source-function, filename, benchmark, source-path, or manifest-label
dispatch was found in `kernel_match.py` or `kernel_match_rewrite.py`.

All 20 normalized fixture IR files were copied outside the repository, given
anonymous filenames, and had their `func.func` symbols changed from
`@mfem_*` to unrelated names.  For every fixture:

- the dry-run semantic match report was unchanged;
- the rewritten `kernel.launch` count was unchanged; and
- the rewritten library-symbol multiset was unchanged.

Result: **20/20 rename-invariance checks passed**.  The same dry-run check
passed for all 16 currently retained application-extraction IR artifacts.

As a negative control, changing one contraction's scalar update from
`arith.addf` to `arith.subf` in the anonymized Mass 2D IR reduced the result
from three matches to two.  This confirms that the matcher responds to the
mathematical body rather than accepting the fixture wholesale.

The MFEM contraction route is structurally driven by:

- zero-initialization followed by `out + input0 * input1` reduction semantics;
- Linalg input/output arity and iterator roles;
- indexing maps and reduction/output modes;
- tensor ranks and FP64 element types; and
- physical submap/broadcast legality.

The resulting `cutensornetContraction2_f64` name is an internal library opcode
selected after semantic recognition, not a name derived from an MFEM symbol.

## Live matcher result and stale artifacts

The live matcher recognizes at least one contraction stage in **18 of 20**
normalized semantic kernels.  The two unmatched kernels are the scalarized 2D
and 3D elasticity quadrature functions.

The live per-kernel stage-launch counts are:

| Family | 2D | 3D |
|---|---:|---:|
| convection apply | 5 | 10 |
| curl-curl apply | 5 | 29 |
| diffusion apply | 6 | 14 |
| div-div apply | 6 | 12 |
| elasticity qpoint | 0 | 0 |
| integrate gradient | 4 | 9 |
| integrate value | 1 | 2 |
| interpolate gradient | 4 | 8 |
| interpolate value | 2 | 3 |
| mass apply | 3 | 5 |

Total: **128 FP64 cuTensorNet contraction launches**.

This is not a count of 128 MFEM kernels and not evidence that 18 complete
operators were replaced.  It is a count of small matched contraction regions
inside 20 normalized kernels.  Residual pointwise, coefficient, reduction,
packing, and accumulation stages remain.

The checked-in artifacts disagree with the live matcher:

- `match_results/summary.csv`: 35 launches;
- the generated heading in `match_results/SUMMARY.md`: 35 launches;
- `MATCHING.md`: 98 launches; and
- the live matcher: 128 launches.

The larger 11-row application manifest similarly records 37 matches, while
the live matcher reports 284 candidates.  Navier alone grows from 4 recorded
matches to 130 live candidates.  These expanded counts are analysis-only
until rewritten IR, ABI lowering, compilation, correctness, and silicon
execution are rerun from one pinned tree.

## Correctness and semantic risks

The original-versus-normalized host validation was rerun during this audit.
All 20 semantic pairs passed; the largest observed absolute difference was
`3.109e-15`.

Important remaining risks are:

1. **Resolved after the initial audit:** the zero-plus-contraction rewrite now
   requires SSA dependence from the contraction destination back to the zero
   initializer through storage-preserving tensor views and functional
   writebacks.  The unrelated-zero negative test and the positive direct/view
   cases pass.  The 20-fixture recount remains 18/20 kernels and 128 static
   contraction sites, so none of those sites depended on the rejected bug.
2. Floating-point matching permits commutation and reassociation.  This is a
   real-algebra/tolerance equivalence, not a bitwise-equivalence guarantee.  It
   should either require an explicit reassociation policy or be reported with
   numerical error bounds.
3. The broad rank-generic contraction rule lacks a sufficient near-miss suite.
   Test nonzero accumulators, illegal output/reduction maps, duplicate modes,
   affine rather than pure-dimension maps, aliased views, and unsupported
   physical broadcasts.
4. Existing standalone native-GPU tests use constant-valued basis matrices and
   compare aggregate summaries.  Use nonconstant basis/gradient tensors,
   multiple seeds, multiple element counts/shapes, and direct elementwise
   comparisons against upstream MFEM on identical buffers.

## Extraction and execution audit

The corpus is not unmodified upstream MFEM.  It consists of concrete,
specialized C extractions at `D1D=4`, `Q1D=5`, and FP64.  The raisable variants
manually apply scratch slicing, stage slicing, scalarization, component
packing, and staging.  Several larger paths use handwritten routines from
`application_extractions/missing_stage_kernels.c`.

The 11 larger rows must therefore be described as **MFEM-derived extracted and
manually normalized operator paths**, not complete MFEM applications or an
untouched whole-program transformation.

Historical notes describe compiler-generated residual GPU kernels and composed
networks.  The current tree does not contain the documented
`--prepare-gpu-residual-pipeline` route, and the current composition summaries
contain zero composed networks.  The default retained build path lowers
residual Linalg to loops.  The historical generated-residual and fully
composed claims are not reproducible from the current artifact set and must be
restored with exact scripts/artifacts or retracted from current-facing pages.

## Performance audit

The existing numbers are not yet apples-to-apples:

- native MFEM uses resident CUDA buffers and specialized fused kernels;
- the recorded raised path uses mapped host buffers and includes residual
  stages and different host/device boundaries;
- only 2 of 11 larger paths have an exact native-MFEM CUDA operator baseline;
- only one path has a five-process raised timing campaign, and its native GPU
  number came from a different session;
- retained native and raised harnesses compute different statistics
  (best individual call versus an aggregate mean), despite notes describing a
  shared protocol; and
- complete local raw logs, exact commands, and hashes are not retained for all
  headline rows.

No current MFEM result should be labelled a same-system speedup until both
implementations use the publication protocol in
`PUBLICATION_BENCHMARK_METHODOLOGY.md` on one board with identical input,
residency, synchronization, and timing boundaries.

## Prioritized improvements

### P0 — make the evidence internally consistent

1. Pin a clean compiler tree and regenerate raising, debufferization, matching,
   rewriting, ABI lowering, build, correctness, and silicon artifacts together.
2. Generate prose and viewer totals from the regenerated CSV rather than
   hardcoding historical counts.  Archive stale `_partial` and superseded
   artifacts so they cannot be mistaken for current output.
3. Restore a reproducible residual-GPU pipeline or remove the current-facing
   generated-residual claims.

### P1 — make matcher coverage defensible

1. Commit rename-invariance tests for a standalone fixture and a larger path.
2. Add the zero-buffer provenance/alias proof and structural negative tests
   listed above.
3. Report four separate quantities: semantic kernels, kernels with at least one
   match, matched regions/library launches, and residual operations/backends.
   Never present `18/20` as complete library replacement.
4. Fully lower, compile, and validate every one of the 128 live candidates
   before promoting the expanded count.

### P1 — make correctness and timing publication quality

1. Compare full outputs directly against upstream MFEM with varied,
   nonconstant inputs and declared `atol`/`rtol`.
2. Build exact resident native-MFEM GPU counterparts for each selected paper
   row; do not substitute component sums or approximate fallbacks.
3. Run native CPU, native GPU, and raised GPU on one selected Orin in rotated
   order using five processes, five warmups, twenty iterations, medians, IQR,
   minimum, and maximum.  Retain commands, raw samples, hashes, clock state,
   temperatures, and software versions.

### P2 — improve coverage and performance

1. Automate the scratch/stage slicing and component scalarization needed to
   reach the current normalized IR from the faithful extraction.
2. Keep unmatched residual stages device resident and fuse pointwise and
   accumulation work with adjacent contractions where legal.
3. Compose compatible contraction chains, remove global intermediate
   materializations/snapshots, and reuse plans/workspaces.
4. Treat the elasticity qpoint kernels as legitimate fused/generated-GPU
   candidates unless a pre-existing external library implements their complete
   semantics; do not force an artificial library mapping.
