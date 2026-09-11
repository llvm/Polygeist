# Egglog single-source matching audit

The scalar-body matcher now uses Egglog as the semantic acceptance authority.
Structural unification only proposes bindings for captured scalars; it cannot
accept a match. A cheap associative/commutative/identity fingerprint may reject
an impossible candidate before saturation, but it cannot accept one.

`scripts/correctness/egglog_library_variants.py` audits both the canonical MLIR
library and every production composition formula. It generates five equivalent
forms of each formula (commuted, left-associated, right-associated, add-zero,
and multiply-one), runs every proof in a separate process, and records wall
timeouts rather than substituting a deterministic algebra implementation.

The 2026-09-03 audit used eight Egglog iterations and a five-second wall limit:

- 286 canonical formulas
- 1,430 proofs
- 1,405 proved equivalent
- 25 timed out
- 0 disproved
- 0 harness errors

The 25 timeouts are all five variants of five large production formulas:

- `conv3d_11tap_semantic` (43 AST nodes)
- `cudnnConvolution2D_25tap` (99 AST nodes)
- the three yields of `hypar_weno_weights_js` (295--319 AST nodes)

The raw measurements are in `issues/egglog_library_variants.csv`.

## Semantic source migration

The MLIR file contains 184 `kernel.defn` operations. Fifty-six definitions
contain 60 scalar yield formulas across 59 `linalg.generic` bodies; 128 are
currently ABI-only declarations. Twenty-two production patterns that genuinely
duplicate a semantic MLIR body now keep only dispatch metadata in Python and
load their formulas and step arity from `kernel_library_phase2.mlir`.

Three same-name patterns were tested and intentionally not classified as exact
duplicates:

- `cublasDcopy_tensor` also recognizes a rank-2 broadcast adapter, while its
  canonical MLIR body is a rank-1 copy.
- `cudnnConvolutionFwd_batched` recognizes an initializer plus contraction,
  while its canonical MLIR definition currently contains only the contraction.
- `cudnnBatchNormalizationForwardInference` currently assigns different input
  positions to scale and data in the matcher and MLIR body.

Those contracts must be aligned before their Python formulas can be removed.
The other active Python formulas correspond to ABI-only definitions or adapter
symbols with no same-name MLIR definition, so they are not second copies yet.
The next migration step is to author real semantic bodies for those definitions,
then replace their Python formulas with metadata-only records in the same way.
