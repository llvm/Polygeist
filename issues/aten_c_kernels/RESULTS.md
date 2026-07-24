# Raising and matching results

Date: 2026-07-21; matcher revalidated 2026-07-24

Pipeline:

```
cgeist --raise-scf-to-affine
polygeist-opt --remove-iter-args --affine-parallelize \
  --raise-affine-to-linalg-pipeline --lower-polygeist-submap
polygeist-opt --linalg-debufferize
kernel_match_rewrite.py
```

Results:

| Kernel | Raise | Linalg ops | Residual loops | Existing GPU match |
| --- | --- | ---: | ---: | --- |
| `aten_add` | pass | 1 | 0 | `cudnnAddTensor_batched` |
| `aten_addmm` | pass | 2 | 0 | `cublasDgemm` |
| `aten_batch_norm` | pass | 1 | 0 | `cudnnBatchNormalizationForwardInference` |
| `aten_conv2d` | pass | 2 | 0 | `cudnnConvolutionFwd_batched` |
| `aten_dot` | pass | 1 | 0 | `cublasDdot` |
| `aten_max_pool2d` | pass | 2 | 0 | `cudnnMaxPoolFwd_batched` |
| `aten_mm` | pass | 2 | 0 | `memset_zero_2D` + `cublasDgemm_simple` |
| `aten_mv` | pass | 1 | 0 | `cublasDgemv` |
| `aten_rms_norm` | pass | 2 | 0 | `rmsnorm_f32_tensor` |
| `aten_softmax` | pass | 3 | 0 | `cudnnSoftmaxForward_tensor` |

Summary:

- C frontend success: 10/10
- Raised to at least one Linalg operation: 10/10
- Fully raised, with no residual affine/SCF loop: 10/10
- Matched to an existing GPU-backed definition: 10/10
- Total raised `linalg.generic` operations: 17
- Total emitted `kernel.launch` operations: 11

The 2026-07-24 rerun used the current matcher/library definitions and produced
the same mappings. The added cuTensorNet separable 3D tensor-product definition
did not add an ATen match: none of these kernels has that definition's rank-6
3D tensor-product signature. Their existing cuBLAS, cuDNN, and custom CUDA
routes remain the appropriate implementations.

These results demonstrate that the numerical C loop bodies are within the
raising and matcher coverage. The poor whole-ATen result is therefore chiefly
an extraction/frontend problem around PyTorch's C++ abstraction layer, not a
failure of the Linalg raiser on these computations.

This experiment stops at `kernel.launch`. The next validation stage is ABI
lowering, Jetson cross-compilation, correctness comparison, and execution on
the attached Orin GPU.

## Second extraction batch

Date: 2026-07-24

Fifteen additional C extractions were tested with the same pipeline:

| Kernel | Linalg ops | Residual loops | Matcher result |
| --- | ---: | ---: | --- |
| `aten_adaptive_avg_pool2d` | 2 | 0 | none |
| `aten_avg_pool2d` | 2 | 0 | none |
| `aten_bmm` | 2 | 0 | none |
| `aten_cross` | 3 | 0 | none |
| `aten_gelu` | 1 | 0 | `gelu_tanh_f32_tensor` |
| `aten_im2col` | 1 | 0 | none (copy legality rejection) |
| `aten_layer_norm` | 3 | 0 | none |
| `aten_mean` | 1 | 0 | none |
| `aten_mse_loss` | 2 | 0 | none |
| `aten_outer` | 2 | 0 | `memset_zero_2D` only |
| `aten_relu` | 1 | 0 | none |
| `aten_silu` | 1 | 0 | none |
| `aten_sum` | 2 | 0 | `memset_zero_1D` only |
| `aten_transpose_copy` | 1 | 0 | none (copy legality rejection) |
| `aten_upsample_nearest2d` | 1 | 0 | none |

Second-batch summary:

- frontend and raising success: 15/15
- completely raised with no residual affine/SCF loops: 15/15
- additional `linalg.generic` operations: 25
- kernels with at least one emitted matcher hit: 3/15
- completely consumed by semantic matches: 1/15
- executable, semantically valid whole-kernel match: 1/15 (`aten_gelu`)
- partial helper matches: 2/15 (`outer` and `sum` initialization)
- rejected unsafe copy candidates: 2/15 (`im2col` and transpose)

The copy false-positives expose a legality gap. `im2col` is a window gather and
transpose has a permuted output indexing map, but the selected runtime shim is
a flat contiguous copy. Rank and scalar-body agreement are therefore
insufficient; copy matching must also prove compatible indexing maps and
contiguous view layout.

Across the first two batches, all 25 extracted kernels raise completely,
producing 42 Linalg operations and zero residual loops.

The copy false-positives described above have now been fixed. A copy match must
prove identical input/output indexing maps and shaped tensor types, and it
rejects a source produced by `polygeist.submap`. Consequently transpose and
im2col remain as correct residual Linalg instead of becoming invalid
`cudaCopy*` launches.

## Third extraction batch

Date: 2026-07-24

Twenty-five additional C extractions were tested with the same pipeline:

| Kernel | Linalg ops | Residual loops | Matcher result |
| --- | ---: | ---: | --- |
| `aten_adaptive_avg_pool3d` | 2 | 0 | none |
| `aten_avg_pool3d` | 2 | 0 | none |
| `aten_binary_cross_entropy` | 0 | 1 | none |
| `aten_channel_shuffle` | 1 | 0 | none |
| `aten_clamp` | 1 | 0 | none |
| `aten_conv1d` | 2 | 0 | none |
| `aten_conv3d` | 2 | 0 | none |
| `aten_conv_transpose2d` | 2 | 0 | none |
| `aten_cumsum` | 1 | 0 | none |
| `aten_elu` | 1 | 0 | none |
| `aten_embedding` | 0 | 2 | none |
| `aten_hardsigmoid` | 1 | 0 | none |
| `aten_hardswish` | 1 | 0 | none |
| `aten_hardtanh` | 1 | 0 | none |
| `aten_l1_loss` | 1 | 0 | none |
| `aten_leaky_relu` | 1 | 0 | none |
| `aten_lerp` | 1 | 0 | none |
| `aten_pixel_shuffle` | 1 | 0 | none (layout-aware copy rejection) |
| `aten_prod` | 1 | 0 | none |
| `aten_reflection_pad2d` | 1 | 0 | none |
| `aten_replication_pad2d` | 1 | 0 | none |
| `aten_sigmoid` | 1 | 0 | none |
| `aten_softplus` | 0 | 1 | none |
| `aten_tanh` | 1 | 0 | none |
| `aten_upsample_bilinear2d` | 1 | 0 | none |

The `--select-func` pass was fixed during this batch to preserve the transitive
symbol dependencies of a selected root function. This keeps declarations such
as `logf`, and helper functions called by the root, instead of producing
dangling symbol references. BCE and softplus therefore complete the pipeline,
but external math calls and control flow prevent their loops from becoming
Linalg. Embedding retains two loops because its data-dependent integer index is
an indirect gather.

Third-batch summary:

- frontend/pipeline success: 25/25
- completely raised with no residual affine/SCF loops: 22/25
- additional Linalg operations: 27
- residual loops: 4 across BCE, embedding, and softplus
- new valid GPU-library matches: 0

## Complete 50-kernel corpus

- frontend/pipeline success: 50/50
- raised completely to Linalg: 47/50
- total Linalg operations: 69
- residual loops: 4
- kernels with a valid matcher hit: 13/50
- emitted launches: 14
- valid whole-computation GPU routes: the original 10 plus GELU
- partial helper routes: output initialization in `outer` and `sum`
- unsafe copy rewrites remaining: 0

All sources and every intermediate/result are stored under this directory;
`results/summary.tsv` is the machine-readable aggregate.
