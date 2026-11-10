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

## Direct upstream CUDA frontend audit

On 2026-09-09, the unchanged upstream
`mfem::internal::SmemPAMassApply2D<4,5>` instantiation was tested directly.
The native CUDA source cross-compiled for AArch64/SM87 and ran correctly on
Orin at `NE=1024`; its final checksum exactly matched the independent expected
value. cgeist also parsed the same CUDA translation unit and recovered the
operator's multiply/add loop body. It did not, however, produce a valid or
executable GPU module: the emitted module contains an illegal LLVM-struct
element inside a memref, and the launched CUDA kernel remains a declaration
separate from the recovered lambda computation. The integrated CUDA object
path is also disabled in the current cgeist build.

This result is frontend emission/lowering blocked, not a direct upstream CUDA
raising success. Commands and exact evidence are retained in
`direct_native_cgeist/README.md` and `direct_native_cgeist/run.sh`.

The same operator has also been extracted into a self-contained native CUDA
fixture that removes MFEM's containers and dispatch scaffolding while retaining
the upstream thread layout, shared-memory algorithm, barriers, tensor indexing,
and accumulation semantics. It cross-compiles for AArch64/SM87 and passes a
complete 16,384-element comparison against an independent CPU reference on
Orin with zero error. See `direct_native_cgeist/isolated_mass2_native.cu` and
`direct_native_cgeist/isolated_mass2_native_result.csv`. This validates the
isolated native fixture only; it does not change the blocked direct-cgeist
status or count as a library match.

Running the existing raising and matching passes on this isolated CUDA fixture
produces one `linalg.generic`, representing only the final length-five dot
reduction. The first three contraction stages remain behind CUDA
thread-strided loops, barriers, and reused shared scratch; the matcher reports
`0/1` bodies matched and structured-region analysis rejects the incomplete
region. See `direct_native_cgeist/RAISING_RESULT.md`. This gives a focused
raising target without changing the previously validated native semantics.
