# PVA semantic matcher fixtures

These are independently authored, plain-C formulations of the five operation
families currently represented by the Polygeist PVA runtime:

- signed 3x3 convolution with unsigned Q8 coefficients;
- normalized 3x3 box filtering;
- normalized 3x3 Gaussian filtering;
- normalized 3x3 bilateral filtering; and
- 256-bin unsigned 8-bit histogram equalization.

They intentionally contain no PVA API calls and their function names are not
matcher inputs.  A future matcher must recognize the computation from its
operations, indexing, reductions, and control flow.  The fixtures are not
copied from the gated PVA Solutions source tree.

Run the native semantic checks and the cgeist/MLIR raising audit with:

```sh
bash scripts/correctness/raise_pva_fixtures.sh /tmp/pva_matcher_raise
```

The generated `summary.csv` distinguishes native execution, frontend
translation, and raising.  A successful raising pass does not by itself mean a
PVA library match: residual loops and each raised `linalg.generic` must still be
classified before implementing structure-based library selection.

## Current raising audit

All five native checks pass, and all five functions pass cgeist translation and
the raising pipeline.  The useful computation-level results are:

- convolution: the 3x3 multiply-accumulate is one two-dimensional reduction;
- box filter: the 3x3 sum is one two-dimensional reduction;
- Gaussian: coefficient construction, normalization, and the 3x3 weighted sum
  raise separately;
- bilateral: the joint spatial/range weighted sum raises as one neighborhood
  reduction; and
- histogram equalization: initialization and some CDF work raise, while the
  indirect histogram update and indexed remap remain loops.

The original audit exposed two correctness blockers.  The cgeist frontend now
looks through integer typedefs when selecting signed versus unsigned casts, so
unsigned 8-bit Gaussian/bilateral pixels use `arith.uitofp` and preserve values
above 127.  A focused compiler regression test covers scalar and pointer loads,
both conversion directions, and an execution-level `200 -> 200.0` case.

The histogram CDF recurrence is still represented by a parallel
`linalg.generic`, which is not a valid prefix-sum lowering.  Histogram
equalization must not be counted as an automatic PVA match until that semantic
issue is corrected and end-to-end output comparison passes.
