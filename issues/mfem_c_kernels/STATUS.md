# MFEM extraction and raising status

Upstream MFEM is pinned at `951cf8886b9c0c33fb36a2f0ede268c8d6a0d8b5`.

The current manifest contains 40 concrete kernels:

- 20 faithful originals: H1 value/gradient interpolation and integration,
  partial-assembly mass, diffusion and convection, elasticity quadrature work,
  2D/3D H(curl) curl-curl, and 2D/3D H(div) div-div.
- 20 normalized variants covering every faithful original. These use scratch
  slicing, explicit contraction stages, component specialization for staggered
  spaces, and scalarized pointwise elasticity outputs.

Current sweep result:

- cgeist frontend: 40/40
- affine/Linalg pipeline: 40/40
- fully raised with no residual affine/scf loops: 20/40
- normalized variants fully raised: 20/20
- emitted Linalg operations: 999
- residual loops in the 20 faithful originals: 154

All normalized variants are numerically equivalent to their originals. Value
and gradient maps compare exactly for the tested inputs. Across complete mass,
diffusion, convection, elasticity, curl-curl, and div-div operators, the largest
observed original-versus-normalized error is `7.2e-15`. Adjoint checks for
value/gradient maps and symmetry checks for mass, diffusion, curl-curl, and
div-div are within `1.2e-14`.

The successful normalization is structural: scratch is indexed by component or
element and survives across explicit stages.  This removes false reuse
dependencies while retaining sum factorization.  It increases temporary
storage, so future lowering should recover GPU shared-memory reuse after the
high-level computation has been recognized.

One compiler defect was isolated while doing this.  See
`problems/remove_iter_args_post_reduction_use.c`: `remove-iter-args` can create
an invalid dominance relation when a reduction result is combined with a load
defined after the reduction.  Splitting the pointwise multiply into its own
stage avoids the defect and better exposes the FEM pipeline.

The planned normalization and source-coverage list is complete. Further work
should focus on making these structural normalizations automatic and lowering
the expanded logical scratch tensors back to reused GPU shared memory after
Linalg recognition.
