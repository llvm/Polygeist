# ATen unresolved CUDA-library matcher audit

This report audits every ATen fixture that does **not** currently end in a complete rewrite backed by genuine library/runtime algorithms. It deliberately distinguishes an exact public library operation from a configured primitive, a constrained subset, and mere building blocks. The row-level CSV is the authoritative artifact.

## Scope and headline

- Unresolved fixtures audited: **379** (the other 219/598 already have a complete genuine library/runtime rewrite).
- Complete generated/custom GPU fallbacks still requiring a true library route: **0**.
- No current match: **280**.
- Partial stage match with residual IR: **99**.
- Residual loops still block whole-operation recognition: **179**.
- A related library primitive is not automatically a legal or profitable replacement. The compiler must prove the constraints recorded for that row.

## Corrected availability classification

- **SUBSET_WITH_CONSTRAINTS**: 145
- **BUILDING_BLOCKS_ONLY**: 131
- **NO_PUBLIC_LIBRARY_EQUIVALENT**: 39
- **EXACT_GRAPH_IF_SUPPORTED**: 38
- **EXACT_FIXED_CALL**: 18
- **EXACT_CONFIGURED_PRIMITIVE**: 8

By closest library:

- **cuDNN**: 117
- **CUB**: 101
- **NPP**: 41
- **none**: 39
- **cuSPARSE**: 35
- **cuRAND**: 19
- **cuTENSOR**: 16
- **cuBLAS**: 5
- **CUDA Runtime**: 3
- **cuSOLVER**: 3

Priority:

- **LOW**: 134
- **MEDIUM**: 110
- **HIGH**: 72
- **NONE**: 38
- **HIGHEST**: 25

By concrete compiler gap:

- **RAISING_THEN_LIBRARY_LOWERING**: 179
- **LEGALITY_SPECIALIZATION_AND_BACKEND**: 52
- **MULTI_CALL_COMPOSITION_NOT_MATCHER_ONLY**: 50
- **SEMANTIC_MATCHER_AND_LIBRARY_BACKEND**: 39
- **NO_LINK_ONLY_LIBRARY_ROUTE**: 35
- **GRAPH_PARTITION_RESIDUAL_THEN_LIBRARY_LOWERING**: 24

Current local backend status:

- **LIBRARY_BACKEND_ABSENT**: 202
- **GENERAL_CUDNN_GRAPH_BACKEND_ABSENT**: 63
- **RELATED_CUDNN_WRAPPERS_PRESENT_NEED_GENERALIZATION**: 54
- **NO_PUBLIC_LIBRARY_BACKEND_POSSIBLE**: 39
- **GENERAL_CUTENSOR_BACKEND_ABSENT_CUTENSORNET_IS_NOT_EQUIVALENT**: 16
- **RELATED_CUBLAS_WRAPPERS_PRESENT_NEED_GENERALIZATION**: 5

## Important corrections to the previous audit

- `acos`, `asin`, `atan`, `acosh`, `asinh`, `atanh`, trigonometric/hyperbolic functions, `mish`, `swish`, and `softplus` are explicit `cutensorOperator_t` values. They are generic cuTENSOR descriptor candidates, not missing CUDA APIs.
- CUB supplies implementations of scans, sorts, reductions, and selection, but most ATen rows are not a one-call equivalence until axis layout, tie/index policy, collisions, and determinism are proven.
- NPP is primarily a fixed-type 1D signal / 2D image API. It is relevant to specialized contiguous cases, not a general arbitrary-rank ATen tensor backend.
- cuRAND having the same distribution name is insufficient for PyTorch equivalence: generator algorithm, seed/offset advancement, and transform reproducibility matter.
- cuDNN graphs are promising for pointwise/reduction formula DAGs, but the backend must validate an execution plan; documentation does not promise every arbitrary graph fuses.

## Library portfolio reviewed

- **cuBLAS/cuBLASLt:** preferred for Level-1/2/3 dense algebra and supported quantized matmul. It does not cover arbitrary elementwise formulas or tensor-axis reductions.
- **cuDNN:** preferred for convolution, regular pooling/resampling, dense softmax, normalization, attention, and supported pointwise/reduction graphs. Graph-plan acceptance and layout/type constraints still require a legality query.
- **cuTENSOR:** preferred for arbitrary-rank affine contraction, permutation, supported unary elementwise operators, and ADD/MUL/MIN/MAX reductions. The repository currently does not have this general backend.
- **cuTensorNet:** reviewed as an alternative for multi-tensor contraction networks. It is not a substitute for general pointwise/reduction lowering, and the repository's fixed cuTensorNet wrappers do not cover arbitrary ATen shapes. Simple contractions are better served by cuTENSOR or cuBLAS; larger contraction graphs may later select cuTensorNet.
- **cuSPARSE:** preferred only where the loop is a standardized SpMV/SpMM/SpGEMM/SDDMM or supported sparse-format conversion. Sparse indexing alone does not make an operation a cuSPARSE call.
- **CUB:** existing NVIDIA template implementations for scan/sort/reduce/select. They require a C++ template backend and frequently multi-call composition.
- **NPP:** useful for specialized contiguous signal or 2D-image cases. It is not treated as a general tensor backend.
- **cuRAND:** useful only when generator-state and sequence compatibility are proven; otherwise it covers merely the random-draw stage.
- **cuSOLVER:** relevant to enclosing dense factorizations, not automatically to extracted pivot/reflection helper loops.
- **cuFFT:** no unresolved fixture is an FFT execution. `fftshift`, conjugation, and symmetry fill helpers are layout/pointwise operations, so cuFFT is not their replacement.
- **CUDA Runtime:** memcpy/memset cover regular contiguous transfers only. Concatenation, padding, gathers, combinations, and overlapping writes need more than a runtime copy.
- **CUTLASS/cuDNN frontend templates:** reviewed as implementation frameworks, not counted as link-only fixed APIs. Selecting them requires code generation/template instantiation and therefore is a different backend strategy.

## Highest-value missing work

- **`aten_avg_pool2d_backward_cpu`** → cuDNN `Resample forward/backward (MAXPOOL/AVGPOOL)` (EXACT_FIXED_CALL, whole). Work: preserve current partial match and partition residual graph; then pool descriptor matcher + generic forward/backward lowering.
- **`aten_avg_pool2d_cpu`** → cuDNN `Resample forward/backward (MAXPOOL/AVGPOOL)` (EXACT_FIXED_CALL, whole). Work: finish raising residual loops; then pool descriptor matcher + generic forward/backward lowering.
- **`aten_avg_pool3d`** → cuDNN `Resample forward/backward (MAXPOOL/AVGPOOL)` (EXACT_FIXED_CALL, whole). Work: pool descriptor matcher + generic forward/backward lowering.
- **`aten_avg_pool3d_backward_cpu`** → cuDNN `Resample forward/backward (MAXPOOL/AVGPOOL)` (EXACT_FIXED_CALL, whole). Work: preserve current partial match and partition residual graph; then pool descriptor matcher + generic forward/backward lowering.
- **`aten_avg_pool3d_cpu`** → cuDNN `Resample forward/backward (MAXPOOL/AVGPOOL)` (EXACT_FIXED_CALL, whole). Work: finish raising residual loops; then pool descriptor matcher + generic forward/backward lowering.
- **`aten_batch_norm_backward_cpu`** → cuDNN `Batch/Layer/Group normalization graph` (EXACT_GRAPH_IF_SUPPORTED, whole for supported normalization; otherwise normalization stages). Work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- **`aten_batch_norm_backward_template_cpu`** → cuDNN `Batch/Layer/Group normalization graph` (EXACT_GRAPH_IF_SUPPORTED, whole for supported normalization; otherwise normalization stages). Work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- **`aten_batch_norm_collect_stats_cpu`** → cuDNN `Batch/Layer/Group normalization graph` (SUBSET_WITH_CONSTRAINTS, whole for supported normalization; otherwise normalization stages). Work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- **`aten_batch_norm_stats_cpu`** → cuDNN `Batch/Layer/Group normalization graph` (SUBSET_WITH_CONSTRAINTS, whole for supported normalization; otherwise normalization stages). Work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- **`aten_bf16_dot_cpu`** → cuBLAS `GEMM/StridedBatchedGEMM/GemmBatched` (EXACT_FIXED_CALL, whole). Work: generalize GEMM matcher and device-resident cuBLAS ABI.
- **`aten_bf16_gemv_trans_cpu`** → cuBLAS `GEMM/StridedBatchedGEMM/GemmBatched` (EXACT_FIXED_CALL, whole). Work: generalize GEMM matcher and device-resident cuBLAS ABI.
- **`aten_bilinear_cpu`** → cuTENSOR `cutensorCreateContraction` (EXACT_CONFIGURED_PRIMITIVE, whole). Work: preserve current partial match and partition residual graph; then iterator-count-independent contraction recognition + generic descriptor lowering.
- **`aten_conv_tbc_backward_cpu`** → cuDNN `Convolution forward/backward-data/backward-filter` (EXACT_FIXED_CALL, whole). Work: convolution descriptor extraction + missing forward/backward wrappers.
- **`aten_conv_tbc_cpu`** → cuDNN `Convolution forward/backward-data/backward-filter` (EXACT_FIXED_CALL, whole). Work: convolution descriptor extraction + missing forward/backward wrappers.
- **`aten_conv_transpose2d`** → cuDNN `Convolution forward/backward-data/backward-filter` (EXACT_FIXED_CALL, whole). Work: convolution descriptor extraction + missing forward/backward wrappers.
- **`aten_conv_transpose3d_cpu`** → cuDNN `Convolution forward/backward-data/backward-filter` (EXACT_FIXED_CALL, whole). Work: convolution descriptor extraction + missing forward/backward wrappers.
- **`aten_conv_transpose3d_grad_weight_cpu`** → cuDNN `Convolution forward/backward-data/backward-filter` (EXACT_FIXED_CALL, whole). Work: convolution descriptor extraction + missing forward/backward wrappers.
- **`aten_cummax_cummin_cpu`** → CUB `DeviceScan/DeviceSegmentedScan` (SUBSET_WITH_CONSTRAINTS, whole for contiguous/segmented associative scans). Work: preserve current partial match and partition residual graph; then scan matcher + CUB template backend + axis specialization.
- **`aten_cumprod_backward_cpu`** → CUB `DeviceScan/DeviceSegmentedScan` (SUBSET_WITH_CONSTRAINTS, whole for contiguous/segmented associative scans). Work: finish raising residual loops; then scan matcher + CUB template backend + axis specialization.
- **`aten_cumprod_cpu`** → CUB `DeviceScan/DeviceSegmentedScan` (SUBSET_WITH_CONSTRAINTS, whole for contiguous/segmented associative scans). Work: preserve current partial match and partition residual graph; then scan matcher + CUB template backend + axis specialization.
- **`aten_depthwise_conv3x3_cpu`** → cuDNN `Convolution forward/backward-data/backward-filter` (EXACT_FIXED_CALL, whole). Work: convolution descriptor extraction + missing forward/backward wrappers.
- **`aten_dyn_quant_matmul_4bit_cpu`** → cuBLAS `cublasLtMatmul` (SUBSET_WITH_CONSTRAINTS, matmul stage). Work: finish raising residual loops; then quantized pattern + pack/layout proof + cuBLASLt backend.
- **`aten_eq`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_flash_attention_backward_cpu`** → cuDNN `SDPA forward/backward graph` (SUBSET_WITH_CONSTRAINTS, whole for supported SDPA). Work: finish raising residual loops; then recognize complete attention graph + cuDNN frontend plan backend.
- **`aten_flash_attention_cpu`** → cuDNN `SDPA forward/backward graph` (SUBSET_WITH_CONSTRAINTS, whole for supported SDPA). Work: finish raising residual loops; then recognize complete attention graph + cuDNN frontend plan backend.
- **`aten_ge`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_group_norm_backward_cpu`** → cuDNN `Batch/Layer/Group normalization graph` (EXACT_GRAPH_IF_SUPPORTED, whole for supported normalization; otherwise normalization stages). Work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- **`aten_group_norm_cpu`** → cuDNN `Batch/Layer/Group normalization graph` (EXACT_GRAPH_IF_SUPPORTED, whole for supported normalization; otherwise normalization stages). Work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- **`aten_gt`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_heaviside`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_host_softmax_backward_cpu`** → cuDNN `Softmax forward/backward` (EXACT_FIXED_CALL, whole). Work: finish raising residual loops; then softmax axis matcher + general resident wrapper.
- **`aten_host_softmax_cpu`** → cuDNN `Softmax forward/backward` (EXACT_FIXED_CALL, whole). Work: finish raising residual loops; then softmax axis matcher + general resident wrapper.
- **`aten_hspmm_cpu`** → cuSPARSE `SpMV/SpMM/SpGEMM/SDDMM` (SUBSET_WITH_CONSTRAINTS, whole for standardized sparse algebra). Work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- **`aten_int4pack_mm_cpu`** → cuBLAS `cublasLtMatmul` (SUBSET_WITH_CONSTRAINTS, whole when supported). Work: finish raising residual loops; then quantized-matmul recognizer + cuBLASLt descriptor/runtime backend.
- **`aten_int8pack_mm_cpu`** → cuBLAS `cublasLtMatmul` (SUBSET_WITH_CONSTRAINTS, whole when supported). Work: preserve current partial match and partition residual graph; then quantized-matmul recognizer + cuBLASLt descriptor/runtime backend.
- **`aten_kron_impl_cpu`** → cuTENSOR `cutensorCreateContraction` (EXACT_CONFIGURED_PRIMITIVE, whole). Work: iterator-count-independent contraction recognition + generic descriptor lowering.
- **`aten_kron_out_cpu`** → cuTENSOR `cutensorCreateContraction` (EXACT_CONFIGURED_PRIMITIVE, whole). Work: iterator-count-independent contraction recognition + generic descriptor lowering.
- **`aten_layer_norm`** → cuDNN `Batch/Layer/Group normalization graph` (EXACT_GRAPH_IF_SUPPORTED, whole for supported normalization; otherwise normalization stages). Work: preserve current partial match and partition residual graph; then normalization semantic matcher + cuDNN graph-plan backend.
- **`aten_layer_norm_backward_cpu`** → cuDNN `Batch/Layer/Group normalization graph` (EXACT_GRAPH_IF_SUPPORTED, whole for supported normalization; otherwise normalization stages). Work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- **`aten_layer_norm_cpu_backend`** → cuDNN `Batch/Layer/Group normalization graph` (EXACT_GRAPH_IF_SUPPORTED, whole for supported normalization; otherwise normalization stages). Work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- **`aten_le`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_log_sigmoid_cpu`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_logcumsumexp_cpu`** → CUB `DeviceScan/DeviceSegmentedScan` (SUBSET_WITH_CONSTRAINTS, whole for contiguous/segmented associative scans). Work: finish raising residual loops; then scan matcher + CUB template backend + axis specialization.
- **`aten_logical_and`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_logical_not_f32`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_logical_or`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_logical_xor`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_lt`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_max_pool1d_cpu`** → CUB `DeviceSegmentedReduce::ArgMax` (SUBSET_WITH_CONSTRAINTS, whole for explicit windows). Work: finish raising residual loops; then preserve the multi-output value/index reduction through debufferization; then lower to segmented ArgMax.
- **`aten_max_pool3d_backward_cpu`** → cuDNN `Resample forward/backward (MAXPOOL/AVGPOOL)` (EXACT_FIXED_CALL, whole). Work: finish raising residual loops; then pool descriptor matcher + generic forward/backward lowering.
- **`aten_max_pool3d_cpu`** → cuDNN `Resample forward/backward (MAXPOOL/AVGPOOL)` (EXACT_FIXED_CALL, whole). Work: finish raising residual loops; then pool descriptor matcher + generic forward/backward lowering.
- **`aten_max_values_cpu`** → cuTENSOR `cutensorCreateReduction` (EXACT_CONFIGURED_PRIMITIVE, whole). Work: preserve current partial match and partition residual graph; then generic reduction matcher + cuTENSOR descriptor lowering.
- **`aten_min_values_cpu`** → cuTENSOR `cutensorCreateReduction` (EXACT_CONFIGURED_PRIMITIVE, whole). Work: preserve current partial match and partition residual graph; then generic reduction matcher + cuTENSOR descriptor lowering.
- **`aten_nan_to_num`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_ne`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_nested_batch_offsets_cpu`** → CUB `DeviceScan/DeviceSegmentedScan` (SUBSET_WITH_CONSTRAINTS, whole for contiguous/segmented associative scans). Work: scan matcher + CUB template backend + axis specialization.
- **`aten_remainder`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_rms_norm`** → cuDNN `Batch/Layer/Group normalization graph` (EXACT_GRAPH_IF_SUPPORTED, whole for supported normalization; otherwise normalization stages). Work: preserve current partial match and partition residual graph; then normalization semantic matcher + cuDNN graph-plan backend.
- **`aten_sampled_addmm_sparse_csr_cpu`** → cuSPARSE `SpMV/SpMM/SpGEMM/SDDMM` (SUBSET_WITH_CONSTRAINTS, whole for standardized sparse algebra). Work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- **`aten_sign`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_signbit`** → cuDNN `Pointwise operation graph` (EXACT_GRAPH_IF_SUPPORTED, whole if every node is supported). Work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- **`aten_slow_conv3d_backward_weight_cpu`** → cuDNN `Convolution forward/backward-data/backward-filter` (EXACT_FIXED_CALL, whole). Work: convolution descriptor extraction + missing forward/backward wrappers.
- **`aten_sparse_addmm_cpu`** → cuSPARSE `SpMV/SpMM/SpGEMM/SDDMM` (SUBSET_WITH_CONSTRAINTS, whole for standardized sparse algebra). Work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- **`aten_sparse_addmv_bsr_cpu`** → cuSPARSE `SpMV/SpMM/SpGEMM/SDDMM` (SUBSET_WITH_CONSTRAINTS, whole for standardized sparse algebra). Work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- **`aten_sparse_addmv_csr_cpu`** → cuSPARSE `SpMV/SpMM/SpGEMM/SDDMM` (SUBSET_WITH_CONSTRAINTS, whole for standardized sparse algebra). Work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- **`aten_sparse_csr_addmm_cpu`** → cuSPARSE `SpMV/SpMM/SpGEMM/SDDMM` (SUBSET_WITH_CONSTRAINTS, whole for standardized sparse algebra). Work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- **`aten_spmm_reduce_arg_cpu`** → cuSPARSE `SpMV/SpMM/SpGEMM/SDDMM` (SUBSET_WITH_CONSTRAINTS, whole for standardized sparse algebra). Work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- **`aten_spmm_reduce_backward_input_arg_cpu`** → cuSPARSE `SpMV/SpMM/SpGEMM/SDDMM` (SUBSET_WITH_CONSTRAINTS, whole for standardized sparse algebra). Work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- **`aten_spmm_reduce_backward_other_arg_cpu`** → cuSPARSE `SpMV/SpMM/SpGEMM/SDDMM` (SUBSET_WITH_CONSTRAINTS, whole for standardized sparse algebra). Work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- **`aten_spmm_reduce_backward_other_cpu`** → cuSPARSE `SpMV/SpMM/SpGEMM/SDDMM` (SUBSET_WITH_CONSTRAINTS, whole for standardized sparse algebra). Work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- **`aten_spmm_reduce_cpu`** → cuSPARSE `SpMV/SpMM/SpGEMM/SDDMM` (SUBSET_WITH_CONSTRAINTS, whole for standardized sparse algebra). Work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- **`aten_sspaddmm_cpu`** → cuSPARSE `SpMV/SpMM/SpGEMM/SDDMM` (SUBSET_WITH_CONSTRAINTS, whole for standardized sparse algebra). Work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- **`aten_sum`** → cuTENSOR `cutensorCreateReduction` (EXACT_CONFIGURED_PRIMITIVE, whole). Work: preserve current partial match and partition residual graph; then generic reduction matcher + cuTENSOR descriptor lowering.
- **`aten_sum_cpu_backend`** → cuTENSOR `cutensorCreateReduction` (EXACT_CONFIGURED_PRIMITIVE, whole). Work: preserve current partial match and partition residual graph; then generic reduction matcher + cuTENSOR descriptor lowering.
- **`aten_trilinear_cpu`** → cuTENSOR `cutensorCreateContraction` (EXACT_CONFIGURED_PRIMITIVE, whole). Work: preserve current partial match and partition residual graph; then iterator-count-independent contraction recognition + generic descriptor lowering.
- **`aten_upsample_bilinear2d`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_bilinear2d_aa_backward_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_bilinear2d_aa_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_bilinear2d_backward_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_bilinear2d_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_linear1d_backward_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_linear1d_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_nearest1d_backward_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_nearest1d_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_nearest2d`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_nearest2d_backward_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_nearest2d_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_nearest3d_backward_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_nearest3d_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_nearest_exact1d_backward_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_nearest_exact1d_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_nearest_exact2d_backward_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_nearest_exact2d_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_nearest_exact3d_backward_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- **`aten_upsample_nearest_exact3d_cpu`** → cuDNN `Resample forward/backward` (SUBSET_WITH_CONSTRAINTS, whole for supported coordinate mode). Work: coordinate-mode proof + resample descriptor lowering.
- **`aten_weight_norm_backward_cpu`** → cuDNN `Batch/Layer/Group normalization graph` (SUBSET_WITH_CONSTRAINTS, whole for supported normalization; otherwise normalization stages). Work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- **`aten_weight_norm_cpu`** → cuDNN `Batch/Layer/Group normalization graph` (SUBSET_WITH_CONSTRAINTS, whole for supported normalization; otherwise normalization stages). Work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.

## Family-by-family, per-kernel appendix

Each entry lists the closest reviewed implementation, the strength of the relationship, coverage, and required work. Exact legality constraints are in [`cuda_library_gap_detailed.csv`](cuda_library_gap_detailed.csv).

### adaptive_pooling (12)

- `aten_adaptive_avg_pool2d_backward_cpu` — cuDNN / `regular Resample/pooling`; **SUBSET_WITH_CONSTRAINTS**; coverage: only divisible regular-window cases; work: finish raising residual loops; then prove regular-window specialization; otherwise no one-call library route.
- `aten_adaptive_avg_pool2d_cpu` — cuDNN / `regular Resample/pooling`; **SUBSET_WITH_CONSTRAINTS**; coverage: only divisible regular-window cases; work: finish raising residual loops; then prove regular-window specialization; otherwise no one-call library route.
- `aten_adaptive_avg_pool3d` — cuDNN / `regular Resample/pooling`; **SUBSET_WITH_CONSTRAINTS**; coverage: only divisible regular-window cases; work: prove regular-window specialization; otherwise no one-call library route.
- `aten_adaptive_avg_pool3d_backward_cpu` — cuDNN / `regular Resample/pooling`; **SUBSET_WITH_CONSTRAINTS**; coverage: only divisible regular-window cases; work: finish raising residual loops; then prove regular-window specialization; otherwise no one-call library route.
- `aten_adaptive_avg_pool3d_cpu` — cuDNN / `regular Resample/pooling`; **SUBSET_WITH_CONSTRAINTS**; coverage: only divisible regular-window cases; work: finish raising residual loops; then prove regular-window specialization; otherwise no one-call library route.
- `aten_adaptive_max_pool1d_cpu` — cuDNN / `regular Resample/pooling`; **SUBSET_WITH_CONSTRAINTS**; coverage: only divisible regular-window cases; work: finish raising residual loops; then prove regular-window specialization; otherwise no one-call library route.
- `aten_adaptive_max_pool2d_backward_cpu` — cuDNN / `regular Resample/pooling`; **SUBSET_WITH_CONSTRAINTS**; coverage: only divisible regular-window cases; work: finish raising residual loops; then prove regular-window specialization; otherwise no one-call library route.
- `aten_adaptive_max_pool2d_cpu` — cuDNN / `regular Resample/pooling`; **SUBSET_WITH_CONSTRAINTS**; coverage: only divisible regular-window cases; work: finish raising residual loops; then prove regular-window specialization; otherwise no one-call library route.
- `aten_adaptive_max_pool3d_backward_cpu` — cuDNN / `regular Resample/pooling`; **SUBSET_WITH_CONSTRAINTS**; coverage: only divisible regular-window cases; work: finish raising residual loops; then prove regular-window specialization; otherwise no one-call library route.
- `aten_adaptive_max_pool3d_cpu` — cuDNN / `regular Resample/pooling`; **SUBSET_WITH_CONSTRAINTS**; coverage: only divisible regular-window cases; work: finish raising residual loops; then prove regular-window specialization; otherwise no one-call library route.
- `aten_adaptive_max_pool3d_legacy_backward_cpu` — cuDNN / `regular Resample/pooling`; **SUBSET_WITH_CONSTRAINTS**; coverage: only divisible regular-window cases; work: finish raising residual loops; then prove regular-window specialization; otherwise no one-call library route.
- `aten_adaptive_max_pool3d_legacy_cpu` — cuDNN / `regular Resample/pooling`; **SUBSET_WITH_CONSTRAINTS**; coverage: only divisible regular-window cases; work: finish raising residual loops; then prove regular-window specialization; otherwise no one-call library route.

### arg_reduction (2)

- `aten_argmax_cpu` — CUB / `DeviceReduce/SegmentedReduce on value-index pairs; sort+RLE for mode`; **BUILDING_BLOCKS_ONLY**; coverage: algorithmic stages; work: CUB template backend + index-aware matcher + composition.
- `aten_argmin_cpu` — CUB / `DeviceReduce/SegmentedReduce on value-index pairs; sort+RLE for mode`; **BUILDING_BLOCKS_ONLY**; coverage: algorithmic stages; work: CUB template backend + index-aware matcher + composition.

### attention (2)

- `aten_flash_attention_backward_cpu` — cuDNN / `SDPA forward/backward graph`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported SDPA; work: finish raising residual loops; then recognize complete attention graph + cuDNN frontend plan backend.
- `aten_flash_attention_cpu` — cuDNN / `SDPA forward/backward graph`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported SDPA; work: finish raising residual loops; then recognize complete attention graph + cuDNN frontend plan backend.

### boolean_reduction (1)

- `aten_allany_dims_cpu` — cuDNN / `pointwise operations + reduction operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if graph accepted; work: extract expression DAG + graph legality/cost check + cuDNN plan lowering.

### categorical_sampling (1)

- `aten_multinomial_with_replacement_cpu` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: finish raising residual loops; then RNG-state proof + cuRAND backend; compose unsupported transforms.

### column_reduction (1)

- `aten_quant_col_offsets_cpu` — CUB / `DeviceHistogram or DeviceReduce`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported binning/reduction; work: histogram matcher + CUB backend + semantic guards.

### complex_layout (4)

- `aten_fft_conjugate_symmetry_cpu` — cuTENSOR / `cutensorPermute with CONJ/IDENTITY`; **SUBSET_WITH_CONSTRAINTS**; coverage: conjugate/permutation stage; work: split pure conjugate/permutation stages; compose remaining work.
- `aten_fftshift_cpu` — cuTENSOR / `cutensorPermute with CONJ/IDENTITY`; **SUBSET_WITH_CONSTRAINTS**; coverage: conjugate/permutation stage; work: split pure conjugate/permutation stages; compose remaining work.
- `aten_ifftshift_cpu` — cuTENSOR / `cutensorPermute with CONJ/IDENTITY`; **SUBSET_WITH_CONSTRAINTS**; coverage: conjugate/permutation stage; work: split pure conjugate/permutation stages; compose remaining work.
- `aten_sgn_complex_scalarized` — cuTENSOR / `cutensorPermute with CONJ/IDENTITY`; **SUBSET_WITH_CONSTRAINTS**; coverage: conjugate/permutation stage; work: split pure conjugate/permutation stages; compose remaining work.

### compound_or_specialized (7)

- `aten_dyn_quant_pack_4bit_weight_cpu` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: finish raising residual loops; then retain raised code or permit a generated/custom GPU kernel.
- `aten_erfinv` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_gcd_i32` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: finish raising residual loops; then retain raised code or permit a generated/custom GPU kernel.
- `aten_kaiser_window` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_lcm_i32` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: finish raising residual loops; then retain raised code or permit a generated/custom GPU kernel.
- `aten_nextafter` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_weight_to_int4pack_cpu` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: finish raising residual loops; then retain raised code or permit a generated/custom GPU kernel.

### convolution (7)

- `aten_conv_tbc_backward_cpu` — cuDNN / `Convolution forward/backward-data/backward-filter`; **EXACT_FIXED_CALL**; coverage: whole; work: convolution descriptor extraction + missing forward/backward wrappers.
- `aten_conv_tbc_cpu` — cuDNN / `Convolution forward/backward-data/backward-filter`; **EXACT_FIXED_CALL**; coverage: whole; work: convolution descriptor extraction + missing forward/backward wrappers.
- `aten_conv_transpose2d` — cuDNN / `Convolution forward/backward-data/backward-filter`; **EXACT_FIXED_CALL**; coverage: whole; work: convolution descriptor extraction + missing forward/backward wrappers.
- `aten_conv_transpose3d_cpu` — cuDNN / `Convolution forward/backward-data/backward-filter`; **EXACT_FIXED_CALL**; coverage: whole; work: convolution descriptor extraction + missing forward/backward wrappers.
- `aten_conv_transpose3d_grad_weight_cpu` — cuDNN / `Convolution forward/backward-data/backward-filter`; **EXACT_FIXED_CALL**; coverage: whole; work: convolution descriptor extraction + missing forward/backward wrappers.
- `aten_depthwise_conv3x3_cpu` — cuDNN / `Convolution forward/backward-data/backward-filter`; **EXACT_FIXED_CALL**; coverage: whole; work: convolution descriptor extraction + missing forward/backward wrappers.
- `aten_slow_conv3d_backward_weight_cpu` — cuDNN / `Convolution forward/backward-data/backward-filter`; **EXACT_FIXED_CALL**; coverage: whole; work: convolution descriptor extraction + missing forward/backward wrappers.

### ctc_loss (2)

- `aten_ctc_loss_backward_cpu` — cuDNN / `CTC loss`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole; work: finish raising residual loops; then CTC matcher + API wrapper.
- `aten_ctc_loss_cpu` — cuDNN / `CTC loss`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole; work: finish raising residual loops; then CTC matcher + API wrapper.

### data_movement (3)

- `aten_combinations_cpu` — CUDA Runtime / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: finish raising residual loops; then shape specialization and multi-call composition.
- `aten_copysign` — CUDA Runtime / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.
- `aten_nested_select_cpu` — CUDA Runtime / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.

### dense_linear_algebra (4)

- `aten_bf16_dot_cpu` — cuBLAS / `GEMM/StridedBatchedGEMM/GemmBatched`; **EXACT_FIXED_CALL**; coverage: whole; work: generalize GEMM matcher and device-resident cuBLAS ABI.
- `aten_bf16_gemv_trans_cpu` — cuBLAS / `GEMM/StridedBatchedGEMM/GemmBatched`; **EXACT_FIXED_CALL**; coverage: whole; work: generalize GEMM matcher and device-resident cuBLAS ABI.
- `aten_int4pack_mm_cpu` — cuBLAS / `cublasLtMatmul`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole when supported; work: finish raising residual loops; then quantized-matmul recognizer + cuBLASLt descriptor/runtime backend.
- `aten_int8pack_mm_cpu` — cuBLAS / `cublasLtMatmul`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole when supported; work: preserve current partial match and partition residual graph; then quantized-matmul recognizer + cuBLASLt descriptor/runtime backend.

### distance (4)

- `aten_cdist_backward_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: finish raising residual loops; then extract and partition expression/stage graph; validate plan or keep raised code.
- `aten_cdist_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: finish raising residual loops; then extract and partition expression/stage graph; validate plan or keep raised code.
- `aten_pdist_backward_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: finish raising residual loops; then extract and partition expression/stage graph; validate plan or keep raised code.
- `aten_pdist_forward_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: finish raising residual loops; then extract and partition expression/stage graph; validate plan or keep raised code.

### histogram_count (4)

- `aten_bincount_cpu` — CUB / `DeviceHistogram or DeviceReduce`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported binning/reduction; work: finish raising residual loops; then histogram matcher + CUB backend + semantic guards.
- `aten_count_nonzero_impl_cpu` — CUB / `DeviceHistogram or DeviceReduce`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported binning/reduction; work: preserve current partial match and partition residual graph; then histogram matcher + CUB backend + semantic guards.
- `aten_histogramdd_cpu` — CUB / `DeviceHistogram or DeviceReduce`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported binning/reduction; work: finish raising residual loops; then histogram matcher + CUB backend + semantic guards.
- `aten_histogramdd_linear_cpu` — CUB / `DeviceHistogram or DeviceReduce`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported binning/reduction; work: finish raising residual loops; then histogram matcher + CUB backend + semantic guards.

### index_generation (7)

- `aten_flatten_indices_launch_cpu` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.
- `aten_nested_to_mask_cpu` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.
- `aten_sparse_flatten_indices_cpu` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.
- `aten_tril_indices_cpu` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: finish raising residual loops; then shape specialization and multi-call composition.
- `aten_triu_indices_cpu` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: finish raising residual loops; then shape specialization and multi-call composition.
- `aten_triu_mask_cpu` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.
- `aten_triu_tril_batch_cpu` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.

### indexed_data_movement (25)

- `aten_embedding` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_gather_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_gather_expanded_index_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_index_copy_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_index_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_index_fill_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_index_put_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_index_put_impl_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_index_select_dim1_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_index_select_out_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_masked_scatter_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_masked_select_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_masked_select_serial_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_nested_where_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_nested_where_out_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_nonzero_out_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_put_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_scatter_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_scatter_fill_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_slow_conv3d_backward_input_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_spdiags_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_spmm_reduce_backward_input_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_take_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_unsafe_index_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_where_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.

### indexed_scatter (3)

- `aten_max_unpool2d_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_max_unpool3d_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_max_unpool_backward_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.

### indexed_scatter_reduce (8)

- `aten_index_reduce_impl_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_masked_scatter_backward_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_scatter_add_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_scatter_add_expanded_index_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_scatter_reduce_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_scatter_reduce_expanded_index_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_scatter_reduce_two_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_scatter_scalar_reduce_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.

### integer_pointwise (6)

- `aten_bitwise_and_i32` — NPP / `signal logical/shift primitives`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole only for flat supported integer signals; work: layout/type specialization + NPP wrapper; retain nonmatching cases.
- `aten_bitwise_not_i32` — NPP / `signal logical/shift primitives`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole only for flat supported integer signals; work: layout/type specialization + NPP wrapper; retain nonmatching cases.
- `aten_bitwise_or_i32` — NPP / `signal logical/shift primitives`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole only for flat supported integer signals; work: layout/type specialization + NPP wrapper; retain nonmatching cases.
- `aten_bitwise_xor_i32` — NPP / `signal logical/shift primitives`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole only for flat supported integer signals; work: layout/type specialization + NPP wrapper; retain nonmatching cases.
- `aten_lshift_i32` — NPP / `signal logical/shift primitives`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole only for flat supported integer signals; work: layout/type specialization + NPP wrapper; retain nonmatching cases.
- `aten_rshift_i32` — NPP / `signal logical/shift primitives`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole only for flat supported integer signals; work: layout/type specialization + NPP wrapper; retain nonmatching cases.

### loss (10)

- `aten_binary_cross_entropy` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: extract and partition expression/stage graph; validate plan or keep raised code.
- `aten_l1_loss` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: extract and partition expression/stage graph; validate plan or keep raised code.
- `aten_multi_margin_loss_backward_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: finish raising residual loops; then extract and partition expression/stage graph; validate plan or keep raised code.
- `aten_multi_margin_loss_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: finish raising residual loops; then extract and partition expression/stage graph; validate plan or keep raised code.
- `aten_multilabel_margin_loss_backward_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: finish raising residual loops; then extract and partition expression/stage graph; validate plan or keep raised code.
- `aten_multilabel_margin_loss_forward_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: finish raising residual loops; then extract and partition expression/stage graph; validate plan or keep raised code.
- `aten_nll_loss2d_backward_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: finish raising residual loops; then extract and partition expression/stage graph; validate plan or keep raised code.
- `aten_nll_loss2d_forward_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: extract and partition expression/stage graph; validate plan or keep raised code.
- `aten_nll_loss_backward_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: finish raising residual loops; then extract and partition expression/stage graph; validate plan or keep raised code.
- `aten_nll_loss_forward_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: finish raising residual loops; then extract and partition expression/stage graph; validate plan or keep raised code.

### matrix_factorization (3)

- `aten_eig_complex_vectors_cpu` — cuSOLVER / `dense eig/LU/QR helper APIs`; **BUILDING_BLOCKS_ONLY**; coverage: factorization or helper stage; work: finish raising residual loops; then recognize enclosing factorization; helper alone is not a cuSOLVER call.
- `aten_reflect_conj_tri_cpu` — cuSOLVER / `dense eig/LU/QR helper APIs`; **BUILDING_BLOCKS_ONLY**; coverage: factorization or helper stage; work: recognize enclosing factorization; helper alone is not a cuSOLVER call.
- `aten_unpack_pivots_cpu` — cuSOLVER / `dense eig/LU/QR helper APIs`; **BUILDING_BLOCKS_ONLY**; coverage: factorization or helper stage; work: finish raising residual loops; then recognize enclosing factorization; helper alone is not a cuSOLVER call.

### nan_ignoring_reduction (1)

- `aten_nansum_cpu` — cuDNN / `pointwise operations + reduction operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if graph accepted; work: preserve current partial match and partition residual graph; then extract expression DAG + graph legality/cost check + cuDNN plan lowering.

### normalization (12)

- `aten_batch_norm_backward_cpu` — cuDNN / `Batch/Layer/Group normalization graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole for supported normalization; otherwise normalization stages; work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- `aten_batch_norm_backward_template_cpu` — cuDNN / `Batch/Layer/Group normalization graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole for supported normalization; otherwise normalization stages; work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- `aten_batch_norm_collect_stats_cpu` — cuDNN / `Batch/Layer/Group normalization graph`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported normalization; otherwise normalization stages; work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- `aten_batch_norm_stats_cpu` — cuDNN / `Batch/Layer/Group normalization graph`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported normalization; otherwise normalization stages; work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- `aten_group_norm_backward_cpu` — cuDNN / `Batch/Layer/Group normalization graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole for supported normalization; otherwise normalization stages; work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- `aten_group_norm_cpu` — cuDNN / `Batch/Layer/Group normalization graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole for supported normalization; otherwise normalization stages; work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- `aten_layer_norm` — cuDNN / `Batch/Layer/Group normalization graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole for supported normalization; otherwise normalization stages; work: preserve current partial match and partition residual graph; then normalization semantic matcher + cuDNN graph-plan backend.
- `aten_layer_norm_backward_cpu` — cuDNN / `Batch/Layer/Group normalization graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole for supported normalization; otherwise normalization stages; work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- `aten_layer_norm_cpu_backend` — cuDNN / `Batch/Layer/Group normalization graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole for supported normalization; otherwise normalization stages; work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- `aten_rms_norm` — cuDNN / `Batch/Layer/Group normalization graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole for supported normalization; otherwise normalization stages; work: preserve current partial match and partition residual graph; then normalization semantic matcher + cuDNN graph-plan backend.
- `aten_weight_norm_backward_cpu` — cuDNN / `Batch/Layer/Group normalization graph`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported normalization; otherwise normalization stages; work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.
- `aten_weight_norm_cpu` — cuDNN / `Batch/Layer/Group normalization graph`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported normalization; otherwise normalization stages; work: finish raising residual loops; then normalization semantic matcher + cuDNN graph-plan backend.

### opaque_special_function (2)

- `aten_erfcx` — none / `no defensible public-library mapping identified`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code; revisit only with new library evidence.
- `aten_log_ndtr` — none / `no defensible public-library mapping identified`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code; revisit only with new library evidence.

### optimizer_update (3)

- `aten_fused_adagrad_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: extract and partition expression/stage graph; validate plan or keep raised code.
- `aten_fused_adam_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: finish raising residual loops; then extract and partition expression/stage graph; validate plan or keep raised code.
- `aten_fused_sgd_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: finish raising residual loops; then extract and partition expression/stage graph; validate plan or keep raised code.

### ordering_selection (11)

- `aten_kthvalue_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: finish raising residual loops; then CUB backend + operation-specific composition.
- `aten_median_indices_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: finish raising residual loops; then CUB backend + operation-specific composition.
- `aten_quick_select_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: finish raising residual loops; then CUB backend + operation-specific composition.
- `aten_searchsorted_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: finish raising residual loops; then CUB backend + operation-specific composition.
- `aten_sort_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: finish raising residual loops; then CUB backend + operation-specific composition.
- `aten_topk_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: finish raising residual loops; then CUB backend + operation-specific composition.
- `aten_unique_bool_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: finish raising residual loops; then CUB backend + operation-specific composition.
- `aten_unique_consecutive_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: finish raising residual loops; then CUB backend + operation-specific composition.
- `aten_unique_dim_impl_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: finish raising residual loops; then CUB backend + operation-specific composition.
- `aten_unique_dim_template_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: finish raising residual loops; then CUB backend + operation-specific composition.
- `aten_unique_sorted_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: finish raising residual loops; then CUB backend + operation-specific composition.

### padding (21)

- `aten_circular_pad_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: specialize compatible image cases; otherwise composition.
- `aten_constant_pad_nd_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: preserve current partial match and partition residual graph; then specialize compatible image cases; otherwise composition.
- `aten_jagged_to_padded_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: finish raising residual loops; then specialize compatible image cases; otherwise composition.
- `aten_nested_from_padded_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: specialize compatible image cases; otherwise composition.
- `aten_nested_pad_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: specialize compatible image cases; otherwise composition.
- `aten_nested_to_padded_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: specialize compatible image cases; otherwise composition.
- `aten_padded_to_jagged_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: finish raising residual loops; then specialize compatible image cases; otherwise composition.
- `aten_reflection_pad1d_backward_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: finish raising residual loops; then specialize compatible image cases; otherwise composition.
- `aten_reflection_pad1d_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: specialize compatible image cases; otherwise composition.
- `aten_reflection_pad2d` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: specialize compatible image cases; otherwise composition.
- `aten_reflection_pad2d_backward_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: finish raising residual loops; then specialize compatible image cases; otherwise composition.
- `aten_reflection_pad2d_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: specialize compatible image cases; otherwise composition.
- `aten_reflection_pad3d_backward_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: finish raising residual loops; then specialize compatible image cases; otherwise composition.
- `aten_reflection_pad3d_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: specialize compatible image cases; otherwise composition.
- `aten_replication_pad1d_backward_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: finish raising residual loops; then specialize compatible image cases; otherwise composition.
- `aten_replication_pad1d_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: specialize compatible image cases; otherwise composition.
- `aten_replication_pad2d` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: specialize compatible image cases; otherwise composition.
- `aten_replication_pad2d_backward_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: finish raising residual loops; then specialize compatible image cases; otherwise composition.
- `aten_replication_pad2d_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: specialize compatible image cases; otherwise composition.
- `aten_replication_pad3d_backward_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: finish raising residual loops; then specialize compatible image cases; otherwise composition.
- `aten_replication_pad3d_cpu` — NPP / `nppiCopy*Border`; **SUBSET_WITH_CONSTRAINTS**; coverage: 2D image constant/replicate border subset; work: specialize compatible image cases; otherwise composition.

### patch_extract_scatter (8)

- `aten_col2im_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_conv3d_columns_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_unfold3d_acc_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_unfold3d_copy_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_unfold3d_zero_acc_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: finish raising residual loops; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_unfold3d_zero_copy_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_unfold_backward_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: preserve current partial match and partition residual graph; then indexed-op semantic matcher + collision proof or reduce-by-key composition.
- `aten_unfolded2d_acc_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: indexed-op semantic matcher + collision proof or reduce-by-key composition.

### pointwise (14)

- `aten_eq` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- `aten_ge` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- `aten_gt` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- `aten_le` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- `aten_log_sigmoid_cpu` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- `aten_logical_and` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- `aten_logical_not_f32` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- `aten_logical_or` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- `aten_logical_xor` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- `aten_lt` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- `aten_ne` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- `aten_remainder` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- `aten_sign` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- `aten_signbit` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.

### pointwise_formula (2)

- `aten_heaviside` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.
- `aten_nan_to_num` — cuDNN / `Pointwise operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if every node is supported; work: provenance-preserving expression DAG extraction + cuDNN graph backend.

### pointwise_reduction_formula (16)

- `aten_addr_elementwise` — cuDNN / `pointwise operations + reduction operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if graph accepted; work: extract expression DAG + graph legality/cost check + cuDNN plan lowering.
- `aten_entr` — cuDNN / `pointwise operations + reduction operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if graph accepted; work: extract expression DAG + graph legality/cost check + cuDNN plan lowering.
- `aten_fractional_max_pool2d_backward_cpu` — cuDNN / `MAXPOOL Resample`; **BUILDING_BLOCKS_ONLY**; coverage: window reduction only; work: finish raising residual loops; then multi-stage composition; not a matcher-only gap.
- `aten_fractional_max_pool2d_cpu` — cuDNN / `MAXPOOL Resample`; **BUILDING_BLOCKS_ONLY**; coverage: window reduction only; work: finish raising residual loops; then multi-stage composition; not a matcher-only gap.
- `aten_fractional_max_pool3d_backward_cpu` — cuDNN / `MAXPOOL Resample`; **BUILDING_BLOCKS_ONLY**; coverage: window reduction only; work: finish raising residual loops; then multi-stage composition; not a matcher-only gap.
- `aten_fractional_max_pool3d_cpu` — cuDNN / `MAXPOOL Resample`; **BUILDING_BLOCKS_ONLY**; coverage: window reduction only; work: finish raising residual loops; then multi-stage composition; not a matcher-only gap.
- `aten_isneginf` — cuDNN / `pointwise operations + reduction operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if graph accepted; work: extract expression DAG + graph legality/cost check + cuDNN plan lowering.
- `aten_isposinf` — cuDNN / `pointwise operations + reduction operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if graph accepted; work: extract expression DAG + graph legality/cost check + cuDNN plan lowering.
- `aten_joint_scaling_cpu` — cuDNN / `pointwise operations + reduction operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if graph accepted; work: extract expression DAG + graph legality/cost check + cuDNN plan lowering.
- `aten_ldexp` — cuDNN / `pointwise operations + reduction operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if graph accepted; work: extract expression DAG + graph legality/cost check + cuDNN plan lowering.
- `aten_linalg_powsum_cpu` — cuDNN / `pointwise operations + reduction operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if graph accepted; work: preserve current partial match and partition residual graph; then extract expression DAG + graph legality/cost check + cuDNN plan lowering.
- `aten_powsum_cpu` — cuDNN / `pointwise operations + reduction operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if graph accepted; work: preserve current partial match and partition residual graph; then extract expression DAG + graph legality/cost check + cuDNN plan lowering.
- `aten_quant_saturation_cpu` — cuDNN / `pointwise operations + reduction operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if graph accepted; work: extract expression DAG + graph legality/cost check + cuDNN plan lowering.
- `aten_sinc` — cuDNN / `pointwise operations + reduction operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if graph accepted; work: extract expression DAG + graph legality/cost check + cuDNN plan lowering.
- `aten_xlog1py` — cuDNN / `pointwise operations + reduction operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if graph accepted; work: extract expression DAG + graph legality/cost check + cuDNN plan lowering.
- `aten_xlogy` — cuDNN / `pointwise operations + reduction operation graph`; **EXACT_GRAPH_IF_SUPPORTED**; coverage: whole if graph accepted; work: extract expression DAG + graph legality/cost check + cuDNN plan lowering.

### pooling (8)

- `aten_avg_pool2d_backward_cpu` — cuDNN / `Resample forward/backward (MAXPOOL/AVGPOOL)`; **EXACT_FIXED_CALL**; coverage: whole; work: preserve current partial match and partition residual graph; then pool descriptor matcher + generic forward/backward lowering.
- `aten_avg_pool2d_cpu` — cuDNN / `Resample forward/backward (MAXPOOL/AVGPOOL)`; **EXACT_FIXED_CALL**; coverage: whole; work: finish raising residual loops; then pool descriptor matcher + generic forward/backward lowering.
- `aten_avg_pool3d` — cuDNN / `Resample forward/backward (MAXPOOL/AVGPOOL)`; **EXACT_FIXED_CALL**; coverage: whole; work: pool descriptor matcher + generic forward/backward lowering.
- `aten_avg_pool3d_backward_cpu` — cuDNN / `Resample forward/backward (MAXPOOL/AVGPOOL)`; **EXACT_FIXED_CALL**; coverage: whole; work: preserve current partial match and partition residual graph; then pool descriptor matcher + generic forward/backward lowering.
- `aten_avg_pool3d_cpu` — cuDNN / `Resample forward/backward (MAXPOOL/AVGPOOL)`; **EXACT_FIXED_CALL**; coverage: whole; work: finish raising residual loops; then pool descriptor matcher + generic forward/backward lowering.
- `aten_max_pool1d_cpu` — CUB / `DeviceSegmentedReduce::ArgMax`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for explicit windows; work: finish raising residual loops; then preserve the multi-output value/index reduction through debufferization; then lower to segmented ArgMax.
- `aten_max_pool3d_backward_cpu` — cuDNN / `Resample forward/backward (MAXPOOL/AVGPOOL)`; **EXACT_FIXED_CALL**; coverage: whole; work: finish raising residual loops; then pool descriptor matcher + generic forward/backward lowering.
- `aten_max_pool3d_cpu` — cuDNN / `Resample forward/backward (MAXPOOL/AVGPOOL)`; **EXACT_FIXED_CALL**; coverage: whole; work: finish raising residual loops; then pool descriptor matcher + generic forward/backward lowering.

### qkv_transform (1)

- `aten_transform_bias_rescale_qkv_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: extract and partition expression/stage graph; validate plan or keep raised code.

### quantized_matrix_multiply (1)

- `aten_dyn_quant_matmul_4bit_cpu` — cuBLAS / `cublasLtMatmul`; **SUBSET_WITH_CONSTRAINTS**; coverage: matmul stage; work: finish raising residual loops; then quantized pattern + pack/layout proof + cuBLASLt backend.

### ragged_softmax (3)

- `aten_nested_softmax_backward_cpu` — CUB / `segmented max/sum reductions plus pointwise transforms`; **BUILDING_BLOCKS_ONLY**; coverage: softmax stages; work: finish raising residual loops; then CUB segmented-reduction backend + multi-stage composition.
- `aten_nested_softmax_cpu` — CUB / `segmented max/sum reductions plus pointwise transforms`; **BUILDING_BLOCKS_ONLY**; coverage: softmax stages; work: finish raising residual loops; then CUB segmented-reduction backend + multi-stage composition.
- `aten_nested_softmax_dropout_cpu` — CUB / `segmented max/sum reductions plus pointwise transforms`; **BUILDING_BLOCKS_ONLY**; coverage: softmax stages; work: finish raising residual loops; then CUB segmented-reduction backend + multi-stage composition.

### random_distribution (14)

- `aten_bernoulli_scalar_cpu` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_bernoulli_tensor_cpu` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_binomial_transform_cpu` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: finish raising residual loops; then RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_digamma` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_dirichlet_transform_cpu` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: finish raising residual loops; then RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_igamma` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_igammac` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_lgamma` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_polygamma` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_random_cpu` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_random_from_to_cpu` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_random_full_64_bits_range_cpu` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_randperm_cpu` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: finish raising residual loops; then RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_trigamma` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: RNG-state proof + cuRAND backend; compose unsupported transforms.

### random_generation (4)

- `aten_poisson_transform_cpu` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: finish raising residual loops; then RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_sample_poisson_transform_cpu` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: finish raising residual loops; then RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_sobol_draw_cpu` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: finish raising residual loops; then RNG-state proof + cuRAND backend; compose unsupported transforms.
- `aten_sobol_fast_forward_cpu` — cuRAND / `uniform/normal/lognormal/Poisson/Sobol generators`; **SUBSET_WITH_CONSTRAINTS**; coverage: random draw stage or whole distribution subset; work: finish raising residual loops; then RNG-state proof + cuRAND backend; compose unsupported transforms.

### reduce_and_compact (1)

- `aten_rowwise_prune_cpu` — CUB / `DeviceSelect or sort/reduce-by-key primitives`; **BUILDING_BLOCKS_ONLY**; coverage: supported indexing stages; work: preserve current partial match and partition residual graph; then indexed-op semantic matcher + collision proof or reduce-by-key composition.

### reduction (9)

- `aten_and_reduce_cpu` — CUB / `DeviceReduce with logical/bitwise operator`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for a flat/segmented supported type; work: CUB reduction backend + boolean/integer semantic matcher.
- `aten_max_values_cpu` — cuTENSOR / `cutensorCreateReduction`; **EXACT_CONFIGURED_PRIMITIVE**; coverage: whole; work: preserve current partial match and partition residual graph; then generic reduction matcher + cuTENSOR descriptor lowering.
- `aten_min_values_cpu` — cuTENSOR / `cutensorCreateReduction`; **EXACT_CONFIGURED_PRIMITIVE**; coverage: whole; work: preserve current partial match and partition residual graph; then generic reduction matcher + cuTENSOR descriptor lowering.
- `aten_norm_cpu` — cuTENSOR / `cutensorCreateReduction plus elementwise stages`; **BUILDING_BLOCKS_ONLY**; coverage: reduction stage; work: preserve current partial match and partition residual graph; then raise stages, partition graph, and lower generic reduction descriptors.
- `aten_std_var_all_cpu` — cuTENSOR / `cutensorCreateReduction plus elementwise stages`; **BUILDING_BLOCKS_ONLY**; coverage: reduction stage; work: preserve current partial match and partition residual graph; then raise stages, partition graph, and lower generic reduction descriptors.
- `aten_std_var_cpu` — cuTENSOR / `cutensorCreateReduction plus elementwise stages`; **BUILDING_BLOCKS_ONLY**; coverage: reduction stage; work: finish raising residual loops; then raise stages, partition graph, and lower generic reduction descriptors.
- `aten_sum` — cuTENSOR / `cutensorCreateReduction`; **EXACT_CONFIGURED_PRIMITIVE**; coverage: whole; work: preserve current partial match and partition residual graph; then generic reduction matcher + cuTENSOR descriptor lowering.
- `aten_sum_cpu_backend` — cuTENSOR / `cutensorCreateReduction`; **EXACT_CONFIGURED_PRIMITIVE**; coverage: whole; work: preserve current partial match and partition residual graph; then generic reduction matcher + cuTENSOR descriptor lowering.
- `aten_vector_norm_out_cpu` — cuTENSOR / `cutensorCreateReduction plus elementwise stages`; **BUILDING_BLOCKS_ONLY**; coverage: reduction stage; work: preserve current partial match and partition residual graph; then raise stages, partition graph, and lower generic reduction descriptors.

### resampling (27)

- `aten_grid_sampler_2d_backward_cpu` — NPP / `nppiResize/nppiRemap`; **SUBSET_WITH_CONSTRAINTS**; coverage: forward 2D image subset; work: finish raising residual loops; then specialize proven-compatible 2D forward cases; no generic one-call route.
- `aten_grid_sampler_2d_cpu` — NPP / `nppiResize/nppiRemap`; **SUBSET_WITH_CONSTRAINTS**; coverage: forward 2D image subset; work: specialize proven-compatible 2D forward cases; no generic one-call route.
- `aten_grid_sampler_2d_fallback_cpu` — NPP / `nppiResize/nppiRemap`; **SUBSET_WITH_CONSTRAINTS**; coverage: forward 2D image subset; work: specialize proven-compatible 2D forward cases; no generic one-call route.
- `aten_grid_sampler_2d_quantized_cpu` — NPP / `nppiResize/nppiRemap`; **SUBSET_WITH_CONSTRAINTS**; coverage: forward 2D image subset; work: specialize proven-compatible 2D forward cases; no generic one-call route.
- `aten_grid_sampler_3d_backward_cpu` — NPP / `nppiResize/nppiRemap`; **SUBSET_WITH_CONSTRAINTS**; coverage: forward 2D image subset; work: finish raising residual loops; then specialize proven-compatible 2D forward cases; no generic one-call route.
- `aten_grid_sampler_3d_cpu` — NPP / `nppiResize/nppiRemap`; **SUBSET_WITH_CONSTRAINTS**; coverage: forward 2D image subset; work: finish raising residual loops; then specialize proven-compatible 2D forward cases; no generic one-call route.
- `aten_upsample_bicubic2d_aa_backward_cpu` — NPP / `nppiResize/nppiRemap`; **SUBSET_WITH_CONSTRAINTS**; coverage: forward 2D image subset; work: finish raising residual loops; then specialize proven-compatible 2D forward cases; no generic one-call route.
- `aten_upsample_bicubic2d_aa_cpu` — NPP / `nppiResize/nppiRemap`; **SUBSET_WITH_CONSTRAINTS**; coverage: forward 2D image subset; work: finish raising residual loops; then specialize proven-compatible 2D forward cases; no generic one-call route.
- `aten_upsample_bicubic2d_backward_cpu` — NPP / `nppiResize/nppiRemap`; **SUBSET_WITH_CONSTRAINTS**; coverage: forward 2D image subset; work: finish raising residual loops; then specialize proven-compatible 2D forward cases; no generic one-call route.
- `aten_upsample_bicubic2d_cpu` — NPP / `nppiResize/nppiRemap`; **SUBSET_WITH_CONSTRAINTS**; coverage: forward 2D image subset; work: finish raising residual loops; then specialize proven-compatible 2D forward cases; no generic one-call route.
- `aten_upsample_lanczos2d_aa_backward_cpu` — NPP / `nppiResize/nppiRemap`; **SUBSET_WITH_CONSTRAINTS**; coverage: forward 2D image subset; work: finish raising residual loops; then specialize proven-compatible 2D forward cases; no generic one-call route.
- `aten_upsample_lanczos2d_aa_cpu` — NPP / `nppiResize/nppiRemap`; **SUBSET_WITH_CONSTRAINTS**; coverage: forward 2D image subset; work: finish raising residual loops; then specialize proven-compatible 2D forward cases; no generic one-call route.
- `aten_upsample_linear1d_backward_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_linear1d_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_nearest1d_backward_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_nearest1d_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_nearest2d` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_nearest2d_backward_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_nearest2d_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_nearest3d_backward_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_nearest3d_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_nearest_exact1d_backward_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_nearest_exact1d_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_nearest_exact2d_backward_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_nearest_exact2d_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_nearest_exact3d_backward_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_nearest_exact3d_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: coordinate-mode proof + resample descriptor lowering.

### reverse (1)

- `aten_flip_cpu` — none / `no direct reverse API`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: leave as residual IR.

### scalar_state_update (1)

- `aten_amp_update_scale_cpu` — none / `no defensible public-library mapping identified`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code; revisit only with new library evidence.

### scan (5)

- `aten_cummax_cummin_cpu` — CUB / `DeviceScan/DeviceSegmentedScan`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for contiguous/segmented associative scans; work: preserve current partial match and partition residual graph; then scan matcher + CUB template backend + axis specialization.
- `aten_cumprod_backward_cpu` — CUB / `DeviceScan/DeviceSegmentedScan`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for contiguous/segmented associative scans; work: finish raising residual loops; then scan matcher + CUB template backend + axis specialization.
- `aten_cumprod_cpu` — CUB / `DeviceScan/DeviceSegmentedScan`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for contiguous/segmented associative scans; work: preserve current partial match and partition residual graph; then scan matcher + CUB template backend + axis specialization.
- `aten_logcumsumexp_cpu` — CUB / `DeviceScan/DeviceSegmentedScan`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for contiguous/segmented associative scans; work: finish raising residual loops; then scan matcher + CUB template backend + axis specialization.
- `aten_nested_batch_offsets_cpu` — CUB / `DeviceScan/DeviceSegmentedScan`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for contiguous/segmented associative scans; work: scan matcher + CUB template backend + axis specialization.

### search (3)

- `aten_binary_search_strided_rightmost_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: finish raising residual loops; then CUB backend + operation-specific composition.
- `aten_lower_bound_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: finish raising residual loops; then CUB backend + operation-specific composition.
- `aten_upper_bound_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: finish raising residual loops; then CUB backend + operation-specific composition.

### segmented_reduction (8)

- `aten_embedding_bag_backward_max_cpu` — CUB / `DeviceSegmentedReduce`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for direct reduction primitive; work: finish raising residual loops; then CUB backend + segment/boundary extraction.
- `aten_embedding_bag_backward_sum_cpu` — CUB / `DeviceSegmentedReduce`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for direct reduction primitive; work: finish raising residual loops; then CUB backend + segment/boundary extraction.
- `aten_embedding_bag_counts_cpu` — CUB / `DeviceSegmentedReduce`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for direct reduction primitive; work: finish raising residual loops; then CUB backend + segment/boundary extraction.
- `aten_embedding_bag_counts_uniq_cpu` — CUB / `DeviceSegmentedReduce`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for direct reduction primitive; work: CUB backend + segment/boundary extraction.
- `aten_embedding_bag_max_cpu` — CUB / `DeviceSegmentedReduce`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for direct reduction primitive; work: finish raising residual loops; then CUB backend + segment/boundary extraction.
- `aten_embedding_bag_per_sample_backward_cpu` — CUB / `DeviceSegmentedReduce`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for direct reduction primitive; work: finish raising residual loops; then CUB backend + segment/boundary extraction.
- `aten_segment_reduce_lengths_backward_cpu` — CUB / `DeviceSegmentedReduce`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for direct reduction primitive; work: finish raising residual loops; then CUB backend + segment/boundary extraction.
- `aten_segment_reduce_lengths_cpu` — CUB / `DeviceSegmentedReduce`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for direct reduction primitive; work: finish raising residual loops; then CUB backend + segment/boundary extraction.

### set_membership (1)

- `aten_isin_default_cpu` — CUB / `DeviceRadixSort/SegmentedRadixSort/Select/RLE`; **BUILDING_BLOCKS_ONLY**; coverage: sort/search/select stages; work: CUB backend + operation-specific composition.

### sobol_state_transform (2)

- `aten_sobol_initialize_cpu` — none / `no defensible public-library mapping identified`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code; revisit only with new library evidence.
- `aten_sobol_scramble_cpu` — none / `no defensible public-library mapping identified`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code; revisit only with new library evidence.

### softmax (2)

- `aten_host_softmax_backward_cpu` — cuDNN / `Softmax forward/backward`; **EXACT_FIXED_CALL**; coverage: whole; work: finish raising residual loops; then softmax axis matcher + general resident wrapper.
- `aten_host_softmax_cpu` — cuDNN / `Softmax forward/backward`; **EXACT_FIXED_CALL**; coverage: whole; work: finish raising residual loops; then softmax axis matcher + general resident wrapper.

### sparse_format (6)

- `aten_coalesce_sparse_cpu` — cuSPARSE / `COO/CSR conversion and sparse sorting/pruning APIs`; **SUBSET_WITH_CONSTRAINTS**; coverage: standard conversion/sort stages; work: finish raising residual loops; then format recognizer + cuSPARSE conversion backend + residual composition.
- `aten_compressed_block_convert_cpu` — cuSPARSE / `COO/CSR conversion and sparse sorting/pruning APIs`; **SUBSET_WITH_CONSTRAINTS**; coverage: standard conversion/sort stages; work: finish raising residual loops; then format recognizer + cuSPARSE conversion backend + residual composition.
- `aten_convert_coo_to_csr_cpu` — cuSPARSE / `COO/CSR conversion and sparse sorting/pruning APIs`; **SUBSET_WITH_CONSTRAINTS**; coverage: standard conversion/sort stages; work: finish raising residual loops; then format recognizer + cuSPARSE conversion backend + residual composition.
- `aten_convert_csr_to_coo_cpu` — cuSPARSE / `COO/CSR conversion and sparse sorting/pruning APIs`; **SUBSET_WITH_CONSTRAINTS**; coverage: standard conversion/sort stages; work: finish raising residual loops; then format recognizer + cuSPARSE conversion backend + residual composition.
- `aten_sparse_coo_to_csr_cpu` — cuSPARSE / `COO/CSR conversion and sparse sorting/pruning APIs`; **SUBSET_WITH_CONSTRAINTS**; coverage: standard conversion/sort stages; work: finish raising residual loops; then format recognizer + cuSPARSE conversion backend + residual composition.
- `aten_sparse_matmul_csr_to_coo_cpu` — cuSPARSE / `COO/CSR conversion and sparse sorting/pruning APIs`; **SUBSET_WITH_CONSTRAINTS**; coverage: standard conversion/sort stages; work: finish raising residual loops; then format recognizer + cuSPARSE conversion backend + residual composition.

### sparse_indexed_elementwise (9)

- `aten_cat_sparse_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: preserve current partial match and partition residual graph; then mixed cuSPARSE+CUB graph composition; not a one-call matcher.
- `aten_dense_sparse_add_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: finish raising residual loops; then mixed cuSPARSE+CUB graph composition; not a one-call matcher.
- `aten_index_select_sparse_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: mixed cuSPARSE+CUB graph composition; not a one-call matcher.
- `aten_permute_sparse_coo_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: mixed cuSPARSE+CUB graph composition; not a one-call matcher.
- `aten_sparse_csr_add_dense_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: finish raising residual loops; then mixed cuSPARSE+CUB graph composition; not a one-call matcher.
- `aten_sparse_dense_intersection_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: mixed cuSPARSE+CUB graph composition; not a one-call matcher.
- `aten_sparse_full_coo_indices_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: finish raising residual loops; then mixed cuSPARSE+CUB graph composition; not a one-call matcher.
- `aten_sparse_matmul_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: finish raising residual loops; then mixed cuSPARSE+CUB graph composition; not a one-call matcher.
- `aten_sparse_matmul_maxnnz_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: finish raising residual loops; then mixed cuSPARSE+CUB graph composition; not a one-call matcher.

### sparse_linear_algebra (12)

- `aten_hspmm_cpu` — cuSPARSE / `SpMV/SpMM/SpGEMM/SDDMM`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for standardized sparse algebra; work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- `aten_sampled_addmm_sparse_csr_cpu` — cuSPARSE / `SpMV/SpMM/SpGEMM/SDDMM`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for standardized sparse algebra; work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- `aten_sparse_addmm_cpu` — cuSPARSE / `SpMV/SpMM/SpGEMM/SDDMM`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for standardized sparse algebra; work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- `aten_sparse_addmv_bsr_cpu` — cuSPARSE / `SpMV/SpMM/SpGEMM/SDDMM`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for standardized sparse algebra; work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- `aten_sparse_addmv_csr_cpu` — cuSPARSE / `SpMV/SpMM/SpGEMM/SDDMM`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for standardized sparse algebra; work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- `aten_sparse_csr_addmm_cpu` — cuSPARSE / `SpMV/SpMM/SpGEMM/SDDMM`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for standardized sparse algebra; work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- `aten_spmm_reduce_arg_cpu` — cuSPARSE / `SpMV/SpMM/SpGEMM/SDDMM`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for standardized sparse algebra; work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- `aten_spmm_reduce_backward_input_arg_cpu` — cuSPARSE / `SpMV/SpMM/SpGEMM/SDDMM`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for standardized sparse algebra; work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- `aten_spmm_reduce_backward_other_arg_cpu` — cuSPARSE / `SpMV/SpMM/SpGEMM/SDDMM`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for standardized sparse algebra; work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- `aten_spmm_reduce_backward_other_cpu` — cuSPARSE / `SpMV/SpMM/SpGEMM/SDDMM`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for standardized sparse algebra; work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- `aten_spmm_reduce_cpu` — cuSPARSE / `SpMV/SpMM/SpGEMM/SDDMM`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for standardized sparse algebra; work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.
- `aten_sspaddmm_cpu` — cuSPARSE / `SpMV/SpMM/SpGEMM/SDDMM`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for standardized sparse algebra; work: finish raising residual loops; then sparse descriptor extraction + cuSPARSE generic-API backend.

### sparse_reduction (4)

- `aten_sparse_csr_reduce_dim0_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: finish raising residual loops; then mixed cuSPARSE+CUB graph composition; not a one-call matcher.
- `aten_sparse_csr_reduce_dim1_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: finish raising residual loops; then mixed cuSPARSE+CUB graph composition; not a one-call matcher.
- `aten_sparse_norm_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: mixed cuSPARSE+CUB graph composition; not a one-call matcher.
- `aten_sparse_sum_backward_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: mixed cuSPARSE+CUB graph composition; not a one-call matcher.

### sparse_softmax (4)

- `aten_sparse_coo_softmax_backward_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: finish raising residual loops; then mixed cuSPARSE+CUB graph composition; not a one-call matcher.
- `aten_sparse_coo_softmax_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: finish raising residual loops; then mixed cuSPARSE+CUB graph composition; not a one-call matcher.
- `aten_sparse_softmax_offsets_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: finish raising residual loops; then mixed cuSPARSE+CUB graph composition; not a one-call matcher.
- `aten_sparse_softmax_pools_cpu` — cuSPARSE / `sparse descriptors plus CUB segmented/indexed primitives`; **BUILDING_BLOCKS_ONLY**; coverage: storage and reduction stages; work: mixed cuSPARSE+CUB graph composition; not a one-call matcher.

### special_function (26)

- `aten_airy_ai` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_bessel_j0` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_bessel_j1` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_bessel_y0` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_bessel_y1` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_chebyshev_polynomial_t` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_chebyshev_polynomial_u` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_chebyshev_polynomial_v` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_chebyshev_polynomial_w` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_hermite_polynomial_h` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_hermite_polynomial_he` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_i0` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_i0e` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_i1` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_i1e` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_laguerre_polynomial_l` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_legendre_polynomial_p` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_modified_bessel_i0` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_modified_bessel_i1` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_modified_bessel_k0` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_modified_bessel_k1` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_ndtri` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_scaled_modified_bessel_k0` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_scaled_modified_bessel_k1` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_spherical_bessel_j0` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.
- `aten_zeta` — none / `no public whole-tensor NVIDIA library operation`; **NO_PUBLIC_LIBRARY_EQUIVALENT**; coverage: none; work: retain raised code or permit a generated/custom GPU kernel.

### statistical_mode (1)

- `aten_mode_cpu` — CUB / `DeviceReduce/SegmentedReduce on value-index pairs; sort+RLE for mode`; **BUILDING_BLOCKS_ONLY**; coverage: algorithmic stages; work: finish raising residual loops; then CUB template backend + index-aware matcher + composition.

### tensor_contraction (11)

- `aten_bilinear_cpu` — cuTENSOR / `cutensorCreateContraction`; **EXACT_CONFIGURED_PRIMITIVE**; coverage: whole; work: preserve current partial match and partition residual graph; then iterator-count-independent contraction recognition + generic descriptor lowering.
- `aten_kron_impl_cpu` — cuTENSOR / `cutensorCreateContraction`; **EXACT_CONFIGURED_PRIMITIVE**; coverage: whole; work: iterator-count-independent contraction recognition + generic descriptor lowering.
- `aten_kron_out_cpu` — cuTENSOR / `cutensorCreateContraction`; **EXACT_CONFIGURED_PRIMITIVE**; coverage: whole; work: iterator-count-independent contraction recognition + generic descriptor lowering.
- `aten_trilinear_cpu` — cuTENSOR / `cutensorCreateContraction`; **EXACT_CONFIGURED_PRIMITIVE**; coverage: whole; work: preserve current partial match and partition residual graph; then iterator-count-independent contraction recognition + generic descriptor lowering.
- `aten_upsample_bilinear2d` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_bilinear2d_aa_backward_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_bilinear2d_aa_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_bilinear2d_backward_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: finish raising residual loops; then coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_bilinear2d_cpu` — cuDNN / `Resample forward/backward`; **SUBSET_WITH_CONSTRAINTS**; coverage: whole for supported coordinate mode; work: coordinate-mode proof + resample descriptor lowering.
- `aten_upsample_trilinear3d_backward_cpu` — NPP / `nppiResize/nppiRemap`; **SUBSET_WITH_CONSTRAINTS**; coverage: forward 2D image subset; work: finish raising residual loops; then specialize proven-compatible 2D forward cases; no generic one-call route.
- `aten_upsample_trilinear3d_cpu` — NPP / `nppiResize/nppiRemap`; **SUBSET_WITH_CONSTRAINTS**; coverage: forward 2D image subset; work: specialize proven-compatible 2D forward cases; no generic one-call route.

### tensor_initialization (8)

- `aten_arange_cpu` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.
- `aten_eye_cpu` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.
- `aten_fill` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.
- `aten_fill_diagonal_cpu` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.
- `aten_linspace` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.
- `aten_logspace_cpu` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.
- `aten_masked_fill_cpu` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.
- `aten_range_out_cpu` — CUB / `cudaMemcpy*/Memset or CUB building blocks`; **BUILDING_BLOCKS_ONLY**; coverage: regular contiguous stages; work: shape specialization and multi-call composition.

### triangular_mask (1)

- `aten_triu_tril_single_cpu` — cuDNN / `pointwise/reduction/matmul operation graph`; **BUILDING_BLOCKS_ONLY**; coverage: arithmetic stages; work: extract and partition expression/stage graph; validate plan or keep raised code.

## Primary API evidence

- [cuTENSOR operator/data types](https://docs.nvidia.com/cuda/cutensor/latest/api/types.html) and [operation descriptors](https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html)
- [cuDNN operation families](https://docs.nvidia.com/deeplearning/cudnn/latest/index.html), [pointwise/reduction](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Pointwise.html), and [graph/runtime-fusion constraints](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html)
- [cuBLAS APIs](https://docs.nvidia.com/cuda/cublas/contents.html)
- [cuSPARSE generic APIs](https://docs.nvidia.com/cuda/cusparse/index.html)
- [CUB device-wide primitives](https://nvidia.github.io/cccl/unstable/cub/api/device.html)
- [cuRAND host API](https://docs.nvidia.com/cuda/curand/host-api-overview.html)
- [NPP signal/image primitives](https://docs.nvidia.com/cuda/npp/)
- [cuTensorNet overview](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/overview.html)
- [cuFFT APIs](https://docs.nvidia.com/cuda/cufft/)

## Interpretation

`EXACT_FIXED_CALL` is the strongest route. `EXACT_CONFIGURED_PRIMITIVE` means the mathematics exists but modes/strides/operators must be synthesized. `EXACT_GRAPH_IF_SUPPORTED` requires graph construction and successful plan validation. `SUBSET_WITH_CONSTRAINTS` is only legal after specialization. `BUILDING_BLOCKS_ONLY` is not a matcher-only fix. `NO_PUBLIC_LIBRARY_EQUIVALENT` means link-only lowering is not available in the reviewed NVIDIA libraries.
