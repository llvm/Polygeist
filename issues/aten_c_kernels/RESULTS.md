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
| `aten_im2col` | 1 | 0 | `cudaCopy6D_f32_tensor` (unsafe false-positive) |
| `aten_layer_norm` | 3 | 0 | none |
| `aten_mean` | 1 | 0 | none |
| `aten_mse_loss` | 2 | 0 | none |
| `aten_outer` | 2 | 0 | `memset_zero_2D` only |
| `aten_relu` | 1 | 0 | none |
| `aten_silu` | 1 | 0 | none |
| `aten_sum` | 2 | 0 | `memset_zero_1D` only |
| `aten_transpose_copy` | 1 | 0 | `cudaCopy2D_f32_tensor` (unsafe false-positive) |
| `aten_upsample_nearest2d` | 1 | 0 | none |

Second-batch summary:

- frontend and raising success: 15/15
- completely raised with no residual affine/SCF loops: 15/15
- additional `linalg.generic` operations: 25
- kernels with at least one semantic matcher hit: 5/15
- completely consumed by semantic matches: 3/15
- executable, semantically valid whole-kernel match: 1/15 (`aten_gelu`)
- partial helper matches: 2/15 (`outer` and `sum` initialization)
- unsafe copy false-positives: 2/15 (`im2col` and transpose)

The copy false-positives expose a legality gap. `im2col` is a window gather and
transpose has a permuted output indexing map, but the selected runtime shim is
a flat contiguous copy. Rank and scalar-body agreement are therefore
insufficient; copy matching must also prove compatible indexing maps and
contiguous view layout.

Across both batches, all 25 extracted kernels raise completely, producing 42
Linalg operations and zero residual loops. The current matcher emits 16
launches across 15 kernels, including partial and unsafe matches as classified
above.
