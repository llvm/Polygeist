# MFEM pending-case resolution — 2026-09-08

## Blocked case: integrate value 3D

Resolved. The two structurally recognized FP64 contraction sites now lower,
cross-compile, execute on Orin, and pass independent full-output correctness at
`NE=1024`.

The failure was not in cuTensorNet matching. A non-identity tensor permutation
was incorrectly classified as a rectangular slice during
`polygeist.submapInverse` lowering. `tensor.insert_slice` preserves source
dimension order and therefore cannot implement that transpose. The compiler
now uses affine elementwise writeback for such permutations, with a focused
regression test.

This raises the correctness result from 17/18 to 18/18 kernels and from 126/128
to 128/128 structurally matched contraction sites. It does not add a new timing
row: the existing four-runtime publication campaign still contains 17 kernels,
and integrate-value 3D must be measured under the same protocol before it is
added to performance graphs.

## Unmatched case: elasticity quadrature 2D/3D

Still legitimately unmatched as an external-library operation after a fresh
structural audit.

The scalarized raised form contains four 2D and nine 3D all-parallel FP64
regions, one per stress-tensor component. Each region combines a 2x2 or 3x3
Jacobian determinant and inverse, divergence, Lamé coefficients, quadrature
weight, and symmetric-gradient terms. The regions have 11 tensor inputs in 2D
and 19 in 3D; they contain pointwise multiply/add/subtract/divide expression
graphs but no reduction or contraction iterator.

- cuTensorNet contraction is inapplicable because these regions do not reduce
  over a tensor mode.
- The configured cuDNN generic graph ABI is FP32 and accepts at most four
  tensor inputs; these regions are FP64 with 11 or 19 inputs.
- cuTENSOR can implement individual unary/binary/trinary fragments, including
  reciprocal and arithmetic composition, but no single operation implements
  the complete elasticity expression. Splitting every component into many
  materialized library calls would manufacture a slow library mapping rather
  than recover an optimized primitive.
- Batched dense inversion can cover only the tiny Jacobian-inverse fragment;
  it does not cover the surrounding stress calculation.

Therefore elasticity should remain reported as a fused/generated-GPU
opportunity and as evidence of a missing vendor-library primitive. It must not
be counted as an external-library match unless a pre-existing API implements
the complete semantics or a principled graph backend is added and validated.
