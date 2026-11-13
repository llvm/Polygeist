# lib_v2: PolyBench matching against netlib-raised specs

Date: 2026-10-27. Specs come from Reference-LAPACK commit `6a1a8b8a`.
Contract: `generic_solver/lib_v2/README.md`.
Written by the parent session from the agent's final summary. The underlying data is in `issues/lib_v2/` (see Artifacts).

## Summary

9 of the 30 PolyBench kernels match using only lib_v2 specs. No Python Term formulas or hand-written recognizers are involved. All 9 pass a numerical end-to-end check against the clang-compiled originals at both the MINI and SMALL datasets: 8 are bit-exact and gemver is within 3.1e-16 relative.

## Raising recipe

The paper's `raised_debufferized.mlir` files were produced by `run_kernel_e2e.sh --debuf`, not by `polygeist_build.sh`. Under that recipe, debufferizing DGEMV never finishes. Following the README, specs and PolyBench both use the flags from steps 1–2 of `polygeist_build.sh`, defined in one place (`scripts/correctness/lib_v2/recipe.py`), and all 30 kernels were re-raised with them. The flags are recorded in the README section "Raising recipe".

## Specs

The 10 routines (DGEMM, DGEMV, DGER, DSCAL, DAXPY, DDOT, DSYRK, DSYR2K, DSYMM, DTRMM) produced 87 specs. The build also picks up the region-matcher agent's 10 DTRSV/DPOTF2 specs.

| Status | Count | Notes |
|---|---|---|
| Fully raised and numerically validated against the C transliteration (random inputs) | 50 | Only these are in `kernel_library_v2.mlir` (51 defns in the file) |
| Partially raised | 27 | Unrolled unit-stride DAXPY/DSCAL/DDOT, most DTRMM, DSYMM SIDE=L, DSYRK/DSYR2K TRANS='T' with general β |
| Raised to wrong semantics | 1 | `dtrmm_lltu` |
| Not validated | 7 | |
| cgeist crash | 2 | Both DGEMM alpha=0 specs |

None of the DTRSV/DPOTF2 specs qualified. `dtrsv_lnu` and `dtrsv_utu` raise to *wrong semantics*: the forward-substitution dependence is lost. This is a soundness bug in the raising pipeline, independent of matching, and it is why every spec is now numerically validated before it enters the library.

Per-spec details are in `generic_solver/lib_v2/RAISE_STATUS.md`.

## Per-kernel results (production vs v2)

| Kernel | Production launches | v2 launches |
|---|---|---|
| gemm | cublasDgemm | netlib_dgemm_nn |
| 2mm | Dgemm, Dgemm_alpha_only, memset_zero_2D | netlib_dgemm_nn_beta0, netlib_dgemm_nn (α bound to 1.0) |
| 3mm | Dgemm_simple ×3, memset_zero_2D ×3 | netlib_dgemm_nn_beta0 ×3 |
| atax | Dgemv, Dgemv_T, memset ×2 | netlib_dgemv_n_beta0, netlib_dgemv_t_beta0 |
| bicg | Dgemv, Dgemv_T, memset ×2 | netlib_dgemv_n_beta0, netlib_dgemv_t_beta0 |
| mvt | Dgemv, Dgemv_T | netlib_dgemv_n_beta1, netlib_dgemv_t_beta1 |
| gesummv | Dgemv ×2, Daxpby, memset ×2 | netlib_dgemv_t_beta0 ×2 |
| gemver | Dger_rank2, Dgemv_alpha ×2, Daxpby | netlib_dgemv_n_beta1, netlib_dgemv_t_beta1, netlib_daxpy_inc |
| doitgen | Dgemv_T_zero | netlib_dgemv_n_beta0 |
| syrk | cublasDsyrk | — (reference TRANS='T' specs only partly raise) |
| syr2k | cublasDsyr2k | — (same) |
| symm | hand-written recognizer | — |
| trmm | hand-written recognizer | — |
| trisolv | hand-written recognizer | — at the Linalg level; `netlib_dtrsv_utn` via the region matcher (not numerically checked here) |
| cholesky | hand-written recognizer (cpu_library only) | — at the Linalg level; `netlib_dpotf2_u` via the region matcher (not numerically checked here) |
| covariance | memset (+ whole-kernel recognizer in cpu_library/raised_gpu) | — |
| correlation | hand-written recognizer | — |
| deriche | memset ×3 | — |
| gramschmidt | Dgemv_T (cpu_library only) | — (its generics stay in memref form) |
| adi, durbin, fdtd-2d, floyd-warshall, heat-3d, jacobi-1d, jacobi-2d, lu, ludcmp, nussinov, seidel-2d | none | none |

Why the remaining kernels don't match:
- gesummv's final `alpha*tmp + beta*y` and gemver's rank-2 update are unmatched, because the reference BLAS has no AXPBY and only a rank-1 DGER.
- Zero-fills would only match α=0 specs, and those specs are rejected because they never read A or x.
- Stencils have no BLAS equivalent.

The per-kernel `report.json` files list, for each unmatched generic, the bindings and permutations tried and the furthest stage the match reached.

## Correctness

For each matched kernel:
1. Inline the v2 defns at the launch sites with `--lower-kernel-launch`.
2. Lower and link into the PolyBench harness.
3. Compare every output with the clang-compiled original at full precision.

All 9 kernels pass at MINI and SMALL. As a negative control, deliberately swapping A and B in gemm's launch fails with error 11.9, so the check does catch wrong bindings.

Separately from matching, multi-root debufferize miscompiles jacobi-2d and heat-3d even when no matching is done.

## Matcher generalisations needed

- The existing matcher binds inputs by position and ignores access patterns. `rewrite_mlir`'s operand order cannot express a C-ABI spec.
- `match_v2.py` reuses `parse_generics`, `encode_body`, `match_composition` and `_replace_python_bodies_with_mlir_library`. It adds generic matching of the whole spec body: it searches input permutations and loop bijections and compares the composed access functions. The row-major/column-major A/B swap (m := nj, n := ni) falls out of this search.
- Three other generic rules: α bound to the literal 1.0, stride bound to 1, and a rank-reducing slice (needed for doitgen).
- The in-tree passes cannot lower production-recipe tensor IR with dynamic sizes. `legalize_tensor_submaps.py` is a test-only legaliser, used only for spec validation and the end-to-end check, never in matching.

## Deviations and assumptions

- Zero-skip guards were removed in DGER/DSYRK/DSYR2K, and the `TEMP.NE.ONE` check in DTRMM (that removal is exact).
- Quick returns on sizes and scalars are dropped; α≠0 is assumed.
- Unit-stride binding of the `_inc` specs is justified by reading the Fortran, not by a test.
- Raising reassociates the T-form reductions; 8 specs differ from the reference by at most 9e-16.

## Limitations

- Egglog accepts reassociation.
- Spec validation uses one input set per spec, not a proof.
- The cuBLAS call path was not executed; only the substitution of the canonical body was checked.
- Leading dimensions are taken from tensor extents.

## Combined run: Linalg-level matching plus the region matcher (parent session, 2026-10-27)

Command: `sweep_polybench.py --out issues/lib_v2/polybench_matched_combined -j 6`. All 30 kernels finish with status ok.
`summary.json` lists only Linalg-level launches. Region-matcher launches appear in each kernel's `report.json` and are spliced into its `matched.mlir`.

| Kernel | v2 launches | Source |
|---|---|---|
| gemm, 2mm, 3mm, atax, bicg, mvt, gesummv, gemver, doitgen | as in the table above | Linalg level; e2e PASS at MINI and SMALL |
| trisolv | `memref.copy b→x` + `netlib_dtrsv_utn` | region matcher |
| cholesky | `netlib_dpotf2_u` (in place) | region matcher |
| syrk | `netlib_dsyrk_ut` (in place) | region matcher |

**12 of the paper's 17 PolyBench matches are reproduced** using netlib-derived specs and no routine-specific matcher code. gesummv and gemver are only partially matched, as noted above.
- **Proved but not emitted:** syr2k and trmm, because their matches need a layout transpose the emitter doesn't produce yet.
- **Not matched:** symm (the DSYMM SIDE=L specs only partially raise), covariance and correlation (there is no reference routine for them).

Correctness of the three region matches: the region-matcher tests run each spec's C source under the found binding and compare with the application's C source. Errors are ≤8.9e-16 for trisolv and cholesky, and syrk also passes. These three were not run through the full PolyBench e2e harness. That harness inlines the *raised* spec body, and `dpotf2_u`'s raised body is unsound (see `issues/region_match/spec_validation.md`).

## Artifacts

All paths are under `/home/arjaiswal/Polygeist/`.
- `generic_solver/lib_v2/`: `ref_c/*.c`, `manifest/*.json`, `raised/`, `RAISE_STATUS.md`, `README.md`
- `generic_solver/kernel_library_v2.mlir`
- `scripts/correctness/lib_v2/`: `recipe.py`, `build_lib_v2.py`, `match_v2.py`, `mlir_lite.py`, `raise_polybench.py`, `sweep_polybench.py`, `e2e_v2.py`, `validate_specs.py`, `legalize_tensor_submaps.py`
- `issues/lib_v2/`: `polybench_raised/`, `polybench_matched/` (`summary.json` and per-kernel `report.json`), `polybench_matched_with_region/`, `region_splice/`, `e2e/{MINI,SMALL}/`
