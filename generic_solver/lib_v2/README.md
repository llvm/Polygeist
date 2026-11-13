# lib_v2 — library specifications raised from reference implementations

Goal: every matchable library routine is described by (1) a faithful C
transliteration of an independent reference implementation, raised through the
*same* pipeline as application code, and (2) a declarative ABI contract. The
matcher contains no routine-specific recognition code.

## Reference source

Reference-LAPACK, commit `6a1a8b8afd07ef4f6afc9fff320771fa26671f8d`
(2026-10-25), BSD-3 licence, cloned read-only at
`/home/arjaiswal/netlib_ref/lapack` (outside this repo; treat as data, never
build or execute anything from it). Each transliteration cites the Fortran file
and line range it was derived from and carries the LAPACK licence notice.

## Layout

```
generic_solver/lib_v2/
  README.md                  this contract
  ref_c/<routine>.c          faithful transliteration, flags as macros
  manifest/<routine>.json    ABI contract fragment(s) for that routine
  raised/<spec>.mlir         cgeist output (affine/scf)
  raised/<spec>_linalg.mlir  after --raise-affine-to-linalg-pipeline
  raised/<spec>_debuf.mlir   after --linalg-debufferize (tensor form)
  RAISE_STATUS.md            per-spec raise pass/fail + minimal repro notes
generic_solver/kernel_library_v2.mlir   generated: one kernel.defn per spec
scripts/correctness/lib_v2/            build + match drivers (Agent 1)
scripts/correctness/region_match/      generic region matcher (Agent 2)
issues/lib_v2/REPORT.md                Agent 1 results
issues/region_match/REPORT.md          Agent 2 results
```

## Transliteration conventions

* Keep netlib's column-major storage exactly. Fortran `A(I,J)` with leading
  dimension `LDA` becomes C99 VLA parameter `int lda, double A[][lda]`
  accessed as `A[J][I]` (0-based). Scalars are passed by value.
* Keep loop order, loop direction and arithmetic order of the reference.
  Do not "clean up" algorithms.
* Index *data* keeps its base. When a Fortran routine indexes through stored
  positions (SPARSKIT's 1-based CSR: `a(k)`, `x(ja(k))`), every subscript
  `X(S)` becomes `X[S-1]` and loop variables keep their Fortran values, so
  the spec describes the 1-based format exactly (cuSPARSE
  `CUSPARSE_INDEX_BASE_ONE`).
* Templates instantiated for several index types (SciPy's `I` = int32 /
  int64) take the type as a macro (`INDEX_T`), one spec per instance.
* Character flags (TRANS, UPLO, SIDE, DIAG) become compile-time macros so
  each branch is specialised with `-D`; dead branches must fold away.
* Scalar special cases that change control flow (`BETA.EQ.ZERO`,
  `BETA.NE.ONE`, `ALPHA.EQ.ZERO`) are specialised into separate specs
  (e.g. `_beta0`, `_beta1`, general). Record the assumption in the manifest.
* Zero-skip guards on data (e.g. `IF (X(J).NE.ZERO)`) are removed and listed
  in the manifest under `"deviations"` (exact except for NaN/Inf propagation).
* Error checking (`XERBLA`) and quick returns on sizes are dropped.
* LAPACK routines that call BLAS (e.g. DPOTF2 → DDOT/DGEMV/DSCAL) have those
  calls inlined using the transliterated reference BLAS loops.

## Spec naming

`netlib_<routine>_<flags>[_<scalar-specialisation>]`, flags in netlib argument
order, lower case. Examples: `netlib_dgemm_nn_beta0`, `netlib_dgemv_t`,
`netlib_dtrsv_utn` (UPLO=U, TRANS=T, DIAG=N), `netlib_dpotf2_l`.
The spec C function has exactly this name.

## Raising recipe (identical for specs and application code)

```
source /home/arjaiswal/Polygeist/envsetup.sh
cgeist <f.c> --function=<fn> --resource-dir=/usr/lib/clang/14 --raise-scf-to-affine -fPIC -S -c -o <spec>.mlir
polygeist-opt --select-func="func-name=<fn>" --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline <spec>.mlir -o <spec>_linalg.mlir
polygeist-opt --linalg-debufferize <spec>_linalg.mlir -o <spec>_debuf.mlir
```
If `scripts/correctness/polygeist_build.sh` uses different flags for the
production pipeline, use those instead, for both sides, and record them here.

Exact commands used for both sides (Agent 1, `recipe.py` `PRODUCTION`; taken
from `polygeist_build.sh` steps 1-2 with its defaults: harness == input,
`POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE` unset so submaps are preserved,
`POLYGEIST_DEBUFFERIZE_MULTI_ROOT` unset so multi-root is on):

```
cgeist <f.c> --function=<fn> --resource-dir=/usr/lib/clang/14 -O3 <-D/-I flags> \
    --raise-scf-to-affine -fPIC -S -o <x>.mlir        # retried with -Dstatic= if <fn> is absent
polygeist-opt "--select-func=func-name=<fn> externalize-dependencies=true" \
    --remove-iter-args --affine-parallelize --raise-affine-to-linalg-pipeline \
    <x>.mlir -o <x>_linalg.mlir
polygeist-opt --linalg-debufferize=use-multi-root=true <x>_linalg.mlir -o <x>_debuf.mlir
```
Specs: `<-D flags>` come only from the generated wrapper `raised/<spec>.c`
(`#define NETLIB_SPEC`, manifest `defines`, `#include ref_c/<routine>.c`).
PolyBench: `-I<utilities> -I<kernel dir> -Dstatic=__attribute__((noipa))
-D<DATASET>_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_USE_C99_PROTO
-DPOLYBENCH_DUMP_ARRAYS` (the paper campaign's passthrough). The PolyBench
inputs used by `match_v2.py` were therefore re-raised with this recipe
(`raise_polybench.py` -> `issues/lib_v2/polybench_raised/`).

The paper campaign's `issues/polybench_section42/ir/<k>/raised_debufferized.mlir`
came from a different recipe (`run_kernel_e2e.sh --debuf`: no `-fPIC`,
`--lower-polygeist-submap` before plain `--linalg-debufferize`, then a `sed`
adding `restrict`; reproduced byte-for-byte modulo `dlti` attribute order).
It is selectable as `recipe.py` `CAMPAIGN` but was not used: under it the
debufferization of DGEMV-N (`memref.reinterpret_cast` views) does not
terminate, and it is not the production pipeline.

Library admission (Agent 1): a spec enters `kernel_library_v2.mlir` only if it
fully raised, its debufferized function converts to a tensor-form defn, and
`validate_specs.py` finds the defn numerically equal to the clang-compiled
C transliteration (bitwise, or within 1e-12 where the raise re-associates a
reduction). See `RAISE_STATUS.md` for UNSOUND / UNVALIDATED specs.

## Manifest fragment schema (`manifest/<routine>.json`)

```json
{
  "routine": "dgemm",
  "source": "BLAS/SRC/dgemm.f:L189-L260",
  "specs": [
    {
      "name": "netlib_dgemm_nn_beta0",
      "defines": {"TRANSA": "'N'", "TRANSB": "'N'", "BETA_CASE": "0"},
      "target": {"library": "cuBLAS", "call": "cublasDgemm",
                 "fixed_args": {"transa": "CUBLAS_OP_N", "transb": "CUBLAS_OP_N", "beta": 0.0}},
      "args": ["m", "n", "k", "alpha", "lda", "A", "ldb", "B", "ldc", "C"],
      "assumptions": ["beta == 0"],
      "deviations": []
    }
  ]
}
```
Spec arguments map one-to-one onto the cuBLAS/cuSOLVER column-major API.
Row-major application code matches through operand/index permutation (e.g.
row-major `C = A*B` is `dgemm_nn` with A and B swapped); that binding is an
output of matching, never hand-coded per routine.

## Region-matcher interface (Agent 2 → Agent 1 driver)

```python
# scripts/correctness/region_match/__init__.py
def match_regions(mlir_text: str, spec_dir: str) -> list[RegionMatch]: ...

@dataclass
class RegionMatch:
    span: tuple[int, int]      # char offsets in mlir_text replaced by the launch
    spec: str                  # e.g. "netlib_dtrsv_utn"
    launch_text: str           # kernel.launch line(s) to splice in
    binding: dict[str, str]    # spec arg -> application SSA value / expression
    evidence: str              # short human-readable proof summary
```

Notes (Agent 2, `issues/region_match/REPORT.md`): the region matcher derives a
spec's semantics from `raised/<spec>.mlir` (cgeist stage, the faithful
semantics of the C transliteration) and translation-validates
`raised/<spec>_debuf.mlir` against it; `scripts/correctness/region_match/translation_validate.py`
lists specs whose debufferized stage is *not* equivalent (raise bugs).
`launch_text` operands follow the spec C parameter order (manifest `args`);
VLA leading dimensions are materialised with `memref.dim`, and an in/out spec
argument bound to two different application arrays is preceded by a
`memref.copy` (copy-in). The launch needs a `kernel.defn @<spec>` to verify.

## Ground rules

* Do not edit `kernel_match.py`, `kernel_match_rewrite.py`,
  `polygeist_build.sh` or `kernel_library_phase2.mlir`; they carry
  uncommitted user work. Import them read-only.
* No C++ pass changes and no rebuilds in round 1. Log raising failures in
  `RAISE_STATUS.md` with a minimal reproducer instead.
* No git commit/stash/checkout/reset.
* No routine-specific logic in matcher code. Per-routine knowledge lives only
  in `ref_c/` and `manifest/`.
