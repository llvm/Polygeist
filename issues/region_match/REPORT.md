# Generic region matcher: trsv and Cholesky without hand-written recognizers

Date: 2026-10-27. Written by the parent session from the agent's final summary (the harness blocks subagents from writing report files). Re-verified by the parent: `tests/run_tests.py` gives ALL PASS (20/20), and grepping non-test matcher code for routine names finds nothing.

## Result

PolyBench trisolv and cholesky are matched to netlib DTRSV/DPOTF2 specs by a routine-agnostic matcher in `scripts/correctness/region_match/`; the old regex recognizer (`_render_dense_factorization_regions`) is not used. The only per-routine artifacts are `generic_solver/lib_v2/ref_c/{dtrsv,dpotf2}.c` and `manifest/{dtrsv,dpotf2}.json`. Running `grep -niE 'trsv|potrf|potf2|cholesky|trisolv'` over the non-test matcher code returns nothing. Every emitted launch is also executed numerically against the spec's C source: the maximum error is ≤1.8e-15, while deliberately wrong variants give errors of 3.6e-3 to 1.07.

## Design

1. **Tensors back to memory.** Every tensor SSA value is treated as a version of a buffer, and `linalg.generic` is given sequential memref semantics. Masked-out iterations become guards, and temporaries are made private per loop iteration. As a result, `extract_slice/insert_slice` IR and `submap/submapInverse` IR for the same computation reduce to the same thing.
2. **Element equations plus a dependence check.** For each output element we produce an affine domain and a formula. Every read is classified as an input, a final value, an accumulator, or an inlined temporary. A read that would see an intermediate value is rejected. Bounds are checked with a small Fourier–Motzkin checker in place of islpy. This view does not depend on loop order or on the shape of the IR.
3. **Equivalence in egglog.** The matcher reuses `Term` and `algebra_rules` from `kernel_match.py`, and adds rules for renaming indices, permuting dimensions between row-major and column-major layout, and triangle ↔ mask. The operand binding is asserted as unions between spec names and application names, and the search over bindings is generic. In-place vs out-of-place is handled generically: X_in=b and X_out=x, with a copy b→x emitted before the launch; when b aliases x the match is flagged and no copy is emitted.
4. **Spec semantics are validated, not trusted.** A spec's meaning is taken from its cgeist (affine) stage. Its debufferized stage is checked against that stage (`translation_validate.py`), which is how the raising bugs below were found.
5. **Launch output** follows the contract: operands are in spec C order, `lda` is given as `memref.dim`, and the span can be spliced directly by `match_v2.py`.

Theory: Feautrier 1991; Barthou, Feautrier & Redon 2002 (equivalence of systems of affine recurrence equations); Shashidhar et al., CC 2005 (ADDG-based equivalence checking); Verdoolaege, Janssens & Bruynooghe, TOPLAS 2012; egg/egglog.

## Tests (20/20 pass)

| Case | Result | Note |
|---|---|---|
| PolyBench trisolv (saved IR, freshly raised IR) | `netlib_dtrsv_utn`, A=L, X_in=b, X_out=x | The saved IR (`extract_slice`) and fresh IR (`submap`) give identical equations |
| trsv, axpy form | `dtrsv_utn` | Different loop orientation from PolyBench |
| trsv with explicit temp | `dtrsv_utn` | |
| trsv with unit diagonal | `dtrsv_utu` (not `utn`) | |
| backward solve, upper | `dtrsv_ltn` | |
| trsv in place (b aliases x) | `dtrsv_utn`, flagged, no copy | |
| PolyBench cholesky (saved, fresh) | `netlib_dpotf2_u`, in place | Row-by-row vs DPOTF2's column-by-column order doesn't matter |
| Cholesky, Crout (column-by-column) | `dpotf2_u` | |
| Cholesky, right-looking | `dpotf2_u` | |
| Cholesky, row-major UᵀU | `dpotf2_l` | |
| NEG: transposed solve `L[j][i]` | not `utn`; correctly `lnn` | A valid solve with the other triangle |
| NEG: swapped divide `L[i][i]/x[i]` | rejected | egglog finds no proof (26 bindings tried) |
| NEG: divide before subtract | rejected | No proof |
| NEG: mask `j<=i` | rejected | Dependence check: x[i] reads its own partly updated value |
| NEG: reads `b[j]` instead of `x[j]` | rejected | Element reads differ |
| NEG: b overlaps x shifted by one | rejected | |
| NEG: LDLᵀ | rejected | |
| NEG: Cholesky reading the wrong triangle | rejected | |

## Metrics

- E-graphs have 55–85 nodes and saturate in ≤4 iterations; egglog takes 18–48 ms.
- A whole region match takes 90–280 ms.
- Loading the specs (85, including the lib_v2 BLAS specs) takes about 5.5 s once per process.

## Raising bugs found by translation validation

9 of the 85 specs have a debufferized stage that is NOT equivalent to their cgeist stage (`spec_validation.md`):

- **Inner bound that decreases with the outer index produces an empty loop.** Affects `dtrsv_unn/unu`, `dpotf2_u/l` and four `dtrmm` variants. Minimal repro: `for j<n: for i<n-j-1` produces a view of size 0.
- **`dtrmm_lltu` drops the alpha scaling of the initial value.** It computes `B + Σ alpha·A·B` instead of `alpha·(B + Σ A·B)`.
- **`dtrsv_utu`, `dtrsv_lnu` and `dtrsv_ltu` mark a dimension parallel even though it carries a dependence.** The result is correct only under sequential semantics, which is why the lib_v2 numeric validation reports UNSOUND on one route and EXACT on the other.

Consequence: PolyBench cholesky matches `dpotf2_u`, so any lowering that inlines the *raised* `dpotf2_u` body would give wrong results. Lower to the library call, or inline the cgeist-stage body, until the raising bugs are fixed.

## Limitations and open work

- Matches that need a layout transpose are proved but not emitted (syr2k, trmm).
- Scalar spec parameters cannot be bound to literals, so code with `alpha=1` doesn't match a general-alpha spec.
- Only additive reductions (`acc ± h`) are supported; strided access (`INCX≠1`) and data-dependent control flow are rejected.
- Equivalence is over real numbers, not bitwise.
- Spec loading isn't cached on disk.

Without any extra code, the matcher also handles PolyBench gemm → `dgemm_nn` with A and B swapped (derived, not hand-coded) and syrk → `dsyrk_ut`; both pass the numeric check.

## Artifacts

- `scripts/correctness/region_match/`: `affine.py`, `egg.py`, `equations.py`, `matcher.py`, `mlir_parse.py`, `symexec.py`, `cli.py`, `translation_validate.py`, `tests/run_tests.py`, `tests/numeric_check.py`, `tests/polybench_sweep.py`
- `issues/region_match/`: `results.json`, `spec_validation.md`, `polybench_sweep.json`
- `generic_solver/lib_v2/ref_c/dtrsv.c`, `ref_c/dpotf2.c`, `manifest/dtrsv.json`, `manifest/dpotf2.json`
