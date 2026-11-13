# Larger MFEM application extraction and raising results

## Summary

- Applications represented: 7
- Concrete operator entries: 11
- Entries with complete declared operator-family coverage: 11/11
- Frontend successes: 11/11
- Raising-pipeline successes: 11/11
- Entries with no residual loops: 11/11
- Raised Linalg operations: 1570
- Residual loops: 0
- Semantic matcher groups: 344
- Emitted candidate `kernel.launch` operations: 224

Coverage and raising are reported separately.  All numerical operator families
declared by these application entries are now present in C, and every extracted
entry reaches loop-free Linalg. Mesh handling, restriction/prolongation, MPI
communication, and data-dependent solver control remain application
orchestration rather than element tensor kernels.

## Per-entry results

- `mtop_iso_elasticity`: interpolation, elasticity quadrature function, and
  integration; 76 Linalg operations, zero loops, 16 matches.
- `dfem_minimal_surface`: interpolation, nonlinear minimal-surface quadrature
  function, and integration; 36 Linalg operations, zero loops, 8 matches.
- `ex35p` H1: diffusion + mass; 94 Linalg operations, zero loops, 19 matches.
- `ex35p` H(curl): curl-curl + H(curl) `VectorFEMass`; 166 Linalg operations
  with zero residual loops and 29 matches.
- `ex35p` H(div): div-div + H(div) `VectorFEMass`; 86 Linalg operations and
  zero residual loops with 12 matches.
- `ex9p`: mass + convection + one complete PCG algebra iteration; 46 Linalg
  operations, zero loops, 10 matches.  The convergence loop remains solver
  control.
- `grad_div`: div-div + H(div) `VectorFEMass`; 86 Linalg operations, zero
  residual loops, and 12 matches.
- `abs_l1_jacobi`: complete mass and diffusion branches remain loop-free with
  24/70 Linalg operations and 5/14 matches.  The completed curl-curl + H(curl)
  mass branch now has 166 Linalg operations, zero residual loops, and 29
  matches.
- `navier_tgv`: vector mass, vector diffusion, nonlinear vector convection,
  pressure diffusion, discrete divergence, and discrete gradient are all
  extracted. Scratch-sliced staging produces 720 Linalg operations, zero
  residual loops, 190 semantic candidates, and 70 emitted library calls. The
  direct-C host and Jetson comparisons pass with `max_abs=max_rel=4.16e-17`.

## Interpretation

The scalar additive-reduction fusion now recognizes a zero-initialized private
scratch reduction followed by `output[index] += scratch`. It seeds the Linalg
reduction directly from the selected output element and preserves projected
output indexing maps while enclosing parallel loops are raised. This closes
the direct-contraction raising gap for H(curl), H(div), grad-div, and the
curl-curl smoother without requiring stage-sliced source rewrites.

The former Navier loops combined several tensor-product stages inside each
vector-component loop.  The raising pass deliberately rejects a loop body
containing multiple dependent `linalg.generic` operations.  The extracted C
now makes those boundaries explicit: it packs one component into contiguous
scratch, invokes the existing scalar interpolation/integration stages, applies
the pointwise physics, and scatters the result back.  Components are statically
specialized, and value/gradient scatters are separate loop nests, so every loop
contains at most one raisable tensor stage.

The matched Jetson graph is correctness evidence, not an optimization result:
70 tiny cuTensorNet calls take 142.37 ms/application versus 0.948 ms for the
independent aarch64 `-O3` reference.  Warm per-call host planning/dispatch
dominates device execution.  Plan caching or fused-stage lowering is required
before this decomposition can be performance competitive.

Matcher results are semantic candidates, not performance claims.  Candidate
launches still require ABI, numerical-correctness, and profitability checks.
