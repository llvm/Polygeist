# MFEM normalized-kernel matcher results

The matcher sweep is reproducible with:

```sh
python3 scripts/correctness/mfem_match_sweep.py
```

It debufferizes every fully raised normalized kernel, runs the matcher in
dry-run mode, and also stores the rewritten `kernel.launch` MLIR.

## Semantic matching result

- normalized kernels tested: 20
- debufferization successes: 20
- matcher process successes: 20
- kernels with one or more matches: 12
- kernels without a match: 8
- matched stage groups: 98
- emitted `kernel.launch` operations: 98

Matched stage symbols:

- 92 `cublasGemmFor1x1Conv`
- 5 `cublasDaxpby`
- 1 `cudnnAddTensor_batched`

No match represents an entire FEM operator. All hits are individual contraction
or pointwise stages inside a larger interpolation/operator/integration graph.

Per-kernel matched stage groups:

- 3D curl-curl: 29
- 3D diffusion: 14
- 3D div-div: 12
- 3D convection: 11
- 3D gradient integration: 9
- 3D gradient interpolation: 8
- 3D mass: 5
- 3D value interpolation: 3
- 3D value integration: 2
- 2D curl-curl: 2
- 2D div-div: 2
- 2D convection: 1

The eight unmatched kernels are 2D value interpolation/integration, 2D
gradient interpolation/integration, 2D mass, 2D diffusion, and 2D/3D
elasticity.

## Executable-library legality audit

Currently executable matches: **0**.

The semantic matcher does not enforce the implemented ABI constraints:

- `cublasGemmFor1x1Conv` lowering requires three rank-4 `f32` tensor bases.
  MFEM uses `f64`, and matched operands include rank-3 and rank-5 tensors.
- `cublasDaxpby` lowering requires four operands (`x`, `y`, alpha, beta) and
  rank-1 `f64` tensors. The MFEM rewrite emits two rank-3 operands.
- `cudnnAddTensor_batched` requires rank-4 `f32`; the MFEM stage is `f64`.

There is also a rewrite defect for several multi-stage 3D matches: generated
tensor-cast SSA names are reused, so injecting canonical definitions or parsing
the rewritten module reports duplicate SSA definitions.

These results are therefore useful candidate matches, not deployable CUDA
library mappings. The matcher needs an ABI/shape/type legality filter before
emission. For these tensor contractions, a general `f64` batched GEMM or
cuTENSOR-style contraction definition is a more natural backend candidate than
the current `f32` 1x1-convolution-specific entry.
