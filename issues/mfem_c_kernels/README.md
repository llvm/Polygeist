# MFEM concrete C raising corpus

This directory stores concrete C versions of numerical kernels extracted from
MFEM.  It is intentionally not a collection of wrappers around MFEM: templates,
`DeviceTensor`, `Reshape`, `MFEM_FORALL`, and backend-selection macros have been
specialized away so that the remaining code is the numerical loop algorithm
seen by a compiler after C++ specialization.

The extraction is pinned to MFEM commit
`951cf8886b9c0c33fb36a2f0ede268c8d6a0d8b5`.  Every entry in `manifest.csv`
records its upstream file and symbol.  Concrete tensor sizes are `D1D=4`,
`Q1D=5`, and `VDIM=2`, using `double`.  Arrays use ordinary row-major C layout.

`original/` preserves faithful, directly recognizable algorithm structure,
including local scratch arrays.  If raising requires a structural rewrite, the
rewrite must be added under `normalized/` rather than replacing the original.

Run the frontend and raising survey with:

```sh
python3 scripts/correctness/mfem_raise_sweep.py
```

Validate every normalized implementation against its faithful original with:

```sh
python3 scripts/correctness/mfem_validate_extractions.py
```

Run library matching on all fully raised normalized kernels with:

```sh
python3 scripts/correctness/mfem_match_sweep.py
```

See `MATCHING.md` for the distinction between semantic stage matches and
currently executable ABI-valid library mappings.

The generated MLIR and logs are placed under
`issues/mfem_c_kernels/results/`, while `summary.csv` records frontend success,
raising success, Linalg operation count, and residual loop count.
