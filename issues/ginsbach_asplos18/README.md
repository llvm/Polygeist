# Ginsbach ASPLOS'18 benchmark reproduction

The paper evaluates all sequential C/C++ programs from SNU NPB 1.0.3 and
Parboil: 10 + 11 = 21 programs.  This directory records Polygeist's structural
recognition audit separately from native execution and end-to-end GPU timing.

Pinned sources used locally:

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

`published_idiom_manifest.csv` is the reproducible denominator for comparison
with the paper.  Its 60 rows are transcribed from the per-program stacked bars
in Figure 16 and validated against the category totals in Table 1: 45 scalar
reductions, 5 histogram reductions, 6 stencils, 1 dense matrix operation and
3 sparse matrix operations.  The publication does not identify source lines,
so the manifest deliberately assigns only a stable program/category ordinal;
it does not fabricate a one-to-one source mapping.  The audit reports these
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

## Structured loop audit (2026-09-04)

The loop-aware Egglog audit represents parent loop domains, affine submap
accesses, reductions and safe producer/consumer fusion.  Its family tests cover
scalar/axis reductions, affine stencils, dense GEMM, indirect histograms, CSR
SpMV and Parboil's JDS SpMV.  Indirect candidates remain analysis-only until
their operand roles, collision semantics and CUDA ABI are validated.

The initial loop-aware audit proved 41 structured regions: 17 reduction-shaped
and 17 stencil-shaped regions, with no reachable GEMM.  The implementation has
since closed the concrete blockers needed by the important benchmark families:

- source-faithful Parboil SGEMM lowers to the FP32 alpha/beta cuBLAS ABI;
- NPB MG `resid` and `psinv` lower to CUDA 3-D stencil implementations;
- Parboil's FP32 stencil lowers to the same seven-tap CUDA implementation;
- Parboil histo and tpacf lower to collision-safe CUDA histogram routes;
- Parboil SpMV lowers its JDS loop and NPB CG lowers its CSR loops; and
- incompatible flat-to-ranked C pointer views no longer make CG's frontend IR
  invalid, while `remove-iter-args` preserves SpMV store dominance.

The generated audit output remains the authority for current whole-corpus
counts; unlike the published manifest, it changes as compiler coverage grows.

The 2026-09-04 whole-corpus run is saved as
`program_summary_2026-09-04.csv`.  It contains 18 executable launches: the
original seven initialization launches plus four CG CSR sites, two MG sites,
and one site each for histo, tpacf, SGEMM, JDS SpMV and the Parboil FP32
stencil.  These are compiler sites, not claimed as a one-to-one subset of the
paper's 60 instances because Figure 16 does not publish source locations.

CUDA silicon compilation was attempted on Orin #2 at `192.0.2.1` through
`configured-host`.  Login succeeds and the CUDA 12 runtime/driver is present, but
the machine currently has no `nvcc`, CUDA toolkit, cuBLAS, or cuDNN development
installation and no Polygeist checkout.  Therefore this run validates source
recognition, ABI lowering, and CPU-reference numerics, but not the newly added
device object on silicon.

The reusable device-runtime smoke test is `ginsbach_cuda_smoke.cu`.  It checks
the MG residual and smoother, saturating histogram, TPACF histogram, JDS and
CSR SpMV, and flat seven-point stencil directly against CUDA allocations.
`ginsbach_runtime_cuda_smoke.c` separately checks the production host-mapped
cuBLAS SGEMM wrapper using heap-backed benchmark operands.
