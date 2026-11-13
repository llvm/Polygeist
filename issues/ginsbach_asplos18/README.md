# Ginsbach ASPLOS'18 benchmark reproduction

The paper evaluates all sequential C/C++ programs from SNU NPB and Parboil:
10 + 11 = 21 programs. The paper does not give release numbers or source
revisions. The author's IDL-Demo screenshot uses a path named
`snu-npb-1.0.3/NPB3.3-SER-C`, but equivalence between that demo input and the
paper's evaluation checkout has not been established. This directory records
Polygeist's structural recognition audit separately from native execution and
end-to-end GPU timing.

Pinned source mirrors used locally (not yet proven identical to the paper):

- SNU NPB 1.0.3 mirror: `third_party/ginsbach-snu-npb`, commit
  `4f2aa1b4d3127dbb3612c8aef24b24c69e83013c`.
- Parboil: `third_party/gpu-parboil`, commit
  `ccf3d3126f1754ca85528722f4ced5894ccb852b`.

Run the structural audit with:

```
/usr/bin/python3 scripts/correctness/ginsbach_asplos18_audit.py
```

The generated CSV files and per-source diagnostics are written under
`/tmp/ginsbach_asplos18_audit` by default.  Native SNU tests use Class S.
Parboil execution additionally requires its separately distributed datasets;
source compilation and structural recognition do not.

`published_idiom_manifest.csv` is the reproducible program/category denominator for comparison
with the paper.  Its 60 rows are transcribed from the per-program stacked bars
in Figure 16 and validated against the category totals in Table 1: 45 scalar
reductions, 5 histogram reductions, 6 stencils, 1 dense matrix operation and
3 sparse matrix operations.  The publication does not identify source lines,
so the manifest deliberately assigns only a stable program/category ordinal;
it does not fabricate a one-to-one source mapping. `ginsbach_60_manifest.csv`
and `polygeist_same_inputs.csv` carry the occurrence-level evidence recovered
after bootstrapping. `scripts/initialize_occurrence_manifests.sh` now refuses
to overwrite those reviewed manifests unless explicitly passed `--force`;
that flag intentionally recreates unresolved skeletons. The audit reports the
published counts beside Polygeist's independently detected and executable
sites.

## Initial result (2026-09-03)

- All ten SNU programs built and ran successfully with native GCC at Class S.
- The structural audit covered all 21 programs and 105 translation units.
- `cgeist` accepted 92/105 units; the Linalg pipeline accepted 83/105.
- 36 units contained raised `linalg.generic` operations.
- The production matcher emitted seven launches, but these were only six
  `memset_zero_2D` calls in BT and one `memset_zero_1D` call in LU.  It did not
  recognize the paper's CG sparse matrix-vector, MG stencil, Parboil SGEMM,
  SpMV, histogram, or reduction idioms in this first unmodified-source run.

This is therefore a useful negative baseline, not yet a performance comparison
with the ASPLOS'18 system.  The important blockers are recorded per source in
the audit output.  In particular, whole-file CG reaches an incompatible
fixed-size memref cast in `cgeist`; MG raises 56 generics but none matches a
currently active GPU library specification; and Parboil SGEMM is blocked by
the old source's C++/STL frontend path before its GEMM can reach the matcher.

Parboil's upstream site distributes datasets separately behind a license
acceptance form.  No dataset license was accepted automatically during this
audit, so Parboil native/runtime validation remains pending.

## Extracted Parboil SGEMM result

Three diagnostic forms are retained under `extracted/`:

- `parboil_sgemm.c`: source-faithful computational extraction.  It clears the
  C++/STL frontend blocker, but raising produces one rank-1 dot-product generic
  inside two residual affine loops, so no GEMM launch is emitted.
- `parboil_sgemm_normalized.c`: separates beta scaling from alpha*A*B
  accumulation.  It raises completely to two generics with no residual loops.
  The FP32 layout-aware matcher now recognizes the resulting rank-3 submap
  views and emits `cublasSgemm_broadcast3d_colmajor_nt_alpha_beta`.
- `parboil_sgemm_shaped.c`: gives the same normalized computation an explicit
  two-dimensional VLA ABI.  It reaches the same executable cuBLAS match.

The source-faithful form leaves a rank-1 dot-product generic inside two loops
followed by a beta/alpha epilogue.  Loop-region extraction now proves this
combined region equivalent to SGEMM and rewrites it to the executable
`cublasSgemm_flat_colmajor_nt_alpha_beta` ABI.  The normalized and shaped
extractions exercise the same downstream FP32 alpha/beta cuBLAS route through
already-shaped Linalg.

## External-library-only structured audit (2026-09-05)

The loop-aware Egglog analysis represents parent loop domains, affine submap
accesses, reductions, and safe producer/consumer fusion. Detection is kept
separate from executable library lowering.

All project-authored computational CUDA implementations and their matcher
routes have been removed. The current audit covers 103 real translation units;
MRI-Q `computeQ.cc` is textually included by `main.c` and is not counted twice.
All 103 units complete both frontend translation and the raising pipeline,
producing 677 `linalg.generic` operations.

The audit emits 24 executable external/platform launch sites: six NPB-BT and
one NPB-LU CUDA memset operations, four NPB-CG cuSPARSE CSR SpMV operations,
three NPB-UA cuBLAS DAXPBY compositions, one Parboil SGEMM cuBLAS operation,
and one Parboil stencil cuDNN operation. The fourteen recognized NPB-UA Ddot
sites are now profitability-gated because their statically proven length-5
workloads are too small for individual library dispatch. Excluding the seven
memory-initialization sites leaves 17 computational launches. The
analysis-only inventory separately reports 69 Egglog-proved structured
regions, 29 reduction-shaped regions, 26 stencil-shaped regions, 12 histogram
candidates, and 6 CSR SpMV candidates. Analysis-only candidates must not be
reported as successful library matches.

The checked-in per-program snapshot is `program_summary_2026-09-05.csv`.

See `SILICON_STATUS.md` for the exact external-only status and permitted
next backends.
