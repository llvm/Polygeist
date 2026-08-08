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

## 2026-08-07 executable revalidation

A fresh 50-kernel sweep supersedes the older matcher counts above:

- frontend/pipeline success: 50/50
- completely raised: 47/50
- 69 Linalg operations and 4 residual loops
- 13 launches in 13 kernels
- 11 valid whole-kernel routes, all passing large-shape Jetson correctness
- one partial route (`sum` initialization)
- one unsafe route (`pixel_shuffle` rank-6 copy), withheld from execution

The eleven executable kernels are add, addmm, batch norm, conv2d, dot, GELU,
max-pool2d, mm, mv, RMSNorm, and softmax. Performance and exact problem sizes
are in `silicon_results/large_problem_comparison.csv`. Raised and resident
CUDA values are warm medians of process runs 2--4 on Jetson Orin `sm_87` in
MAXN mode. The resident baseline uses cuBLAS/cuDNN or a fused CUDA kernel with
device-resident operands; the raised path uses the current registered-host-
pointer ABI.

## 2026-08-07 scalar extraction expansion

Thirty additional fixed-f32 scalar specializations were generated from the
pinned ATen TensorIterator implementations and swept with the same pipeline.

- frontend/pipeline success: 80/80 total
- completely raised: 77/80 total
- total Linalg operations: 99
- residual loops: 4 (unchanged from the 50-kernel corpus)
- matcher launches: 13 in 13 kernels (unchanged)
- new scalar batch: 30/30 completely raised, 30 Linalg operations

The lack of new matcher launches is expected: these are small pointwise
formulas and the current policy does not emit one library call per scalar
operation. Raising coverage increased; whole-kernel library coverage did not.

The pinned 224-file source inventory currently classifies translation units as:

- 37 with at least one standalone-C extraction
- 18 additional TensorIterator numerical-body candidates
- 123 additional explicit-loop candidates
- 15 dispatch-only wrappers
- 31 files with no local numerical body

These are source-file counts, not operator counts or claims of complete
operator coverage within the 37 represented files. The authoritative
per-source accounting is extraction_inventory.csv.

The CE ATen tracker is split into 20-row static pages. numerical.html is the
alphabetical ordering; numerical-correctness.html is correctness-first. Both
orderings contain all 80 kernels across four pages.

## 2026-08-07 exhaustive 224-file completion

This section supersedes the earlier incremental corpus totals.

- Standalone C fixtures: 598
- Pinned translation units accounted: 224/224
- Named iterative bodies accounted: 834/834 (`NEEDS_PORT = 0`)
- Concrete CPU dispatch registrations: 358; 357 extracted and one upstream
  null implementation (`mean_stub`)
- cgeist frontend success: 598/598
- raise-pass success: 598/598
- completed `linalg-debufferize` and matcher: 526/598
- stopped at `linalg-debufferize`: 72/598
- raised Linalg operations in successful pipelines: 434
- residual affine/SCF loops in successful pipelines: 424
- emitted library launches: 89 in 86 fixtures
- distinct matched implementation symbols: 18
- launch element-type/ABI mismatches: 0

The 72 debufferization failures are retained as results, not removed from the
corpus. They are concentrated in nested reductions, arg-reductions, pooled
forward reductions, normalization statistics, selection/sorting, and
scatter-accumulation forms. Their C, frontend MLIR, raised MLIR, and diagnostic
logs remain available in the per-kernel result directories.

During this completion pass, `RemoveIterArgs` was fixed to reject a
distributive rewrite when an allegedly invariant epilogue operand is defined
after the loop and therefore does not dominate the proposed in-loop use. The
alloca fallback now handles that case; the regression is covered by
`test/polygeist-opt/remove-iter-args.mlir`. Closed-form output indices also
remove artificial loop-carried counters from triangular, combinations, and
pair-distance extractions. Consequently the final corpus has no raise failure.

Dot-product matching is now element-type gated. f32 reductions emit
`cublasSdot` and lower to `polygeist_cublas_dot_f32`; f64 reductions emit
`cublasDdot` and lower to `polygeist_cublas_dot_f64`. The exhaustive launch
audit found no remaining f32/f64 ABI mismatch.

The CE ATen tracker now has 30 pages per ordering (20 rows per page):
`numerical.html` is alphabetical and `numerical-correctness.html` is
correctness-first. Both contain all 598 fixtures, with upstream implementation
and standalone-C links.

Executable revalidation fixed the cuDNN definition set, batch-normalization
operand order, FP64 dot ABI, GEMV accumulator beta, rank-N wrapper generation,
and softmax slice-to-base SSA provenance. The resident benchmark source is
`benchmarks/aten_resident_cuda_baseline.cu`; raised correctness uses
`benchmarks/aten_raised_jetson_harness.c`.
