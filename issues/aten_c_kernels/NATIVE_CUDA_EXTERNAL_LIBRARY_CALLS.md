# Native CUDA external-library calls in the ATen fixture census

Pinned PyTorch commit: `d7af122d81a49b1fa7a31ba52bd57c026f092646`.

This audit asks what the native PyTorch CUDA path actually calls. It does not reuse the candidate-library classification from `cuda_library_audit.csv`.

- Fixtures audited: 598
- External-library paths identified: 52
- Paths involving a dynamically linked CUDA library: 28
- Paths involving template-generated CUB/Thrust/CUTLASS code: 26
- Exact registered CUDA counterparts with no external call found: 222
- CUDA-supported paths with no external call site found by reverse audit, but without a pinned exact symbol: 228
- No direct CUDA counterpart in the existing availability census: 96
- Earlier `native_cuda_audit.csv` no-CUDA classifications overturned by source evidence: 16

Library counts overlap when one ATen operation has conditional or mixed backends:
- CUB: 16
- CUTLASS: 4
- Thrust: 10
- cuBLAS: 14
- cuDNN: 9
- cuSPARSE: 6

`CUB`, `Thrust`, and `CUTLASS` denote template-generated device code, not a fixed dynamically linked call. CUDA Runtime allocation/copy calls, compiler math intrinsics, and Thrust utility types such as `thrust::pair` are not counted. No fixture's exact counterpart was found to call cuFFT, cuSOLVER, cuRAND, NCCL, or NPP; helper fixtures adjacent to FFT/factorization code are not the full FFT/factorization operation.

`CONDITIONAL_BACKEND` does not prove that a benchmark shape selected that backend. Rows marked `NO_EXACT_CUDA_COUNTERPART` or `ATEN_COMPOSITION` describe the public CUDA operation used for comparison, not a one-to-one CUDA version of a CPU-only helper. See the CSV for per-kernel sources, symbols, confidence, and caveats.

## Identified kernels

- `aten_addmm`: cuBLAS (LINKED_FIXED_LIBRARY); `aten/src/ATen/native/cuda/Blas.cpp`; CUDA addmm lowers to GEMM/GEMM-and-bias after layout preparation.
- `aten_batch_norm`: cuDNN (CONDITIONAL_BACKEND); `aten/src/ATen/native/Normalization.cpp|aten/src/ATen/native/cudnn/BatchNorm.cpp`; The high-level batch-normalization dispatcher may select cuDNN; native CUDA kernels remain fallback paths.
- `aten_batch_norm_cpu_entry`: cuDNN (CONDITIONAL_BACKEND); `aten/src/ATen/native/Normalization.cpp|aten/src/ATen/native/cudnn/BatchNorm.cpp`; The benchmarked high-level batch-normalization route may select cuDNN; this is not true of every extracted internal stage.
- `aten_bmm`: cuBLAS (LINKED_FIXED_LIBRARY); `aten/src/ATen/native/cuda/Blas.cpp`; CUDA bmm uses batched GEMM for the supported dense layouts.
- `aten_coalesce_sparse_cpu`: Thrust (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/sparse/cuda/SparseCUDATensor.cu`; Sparse COO coalescing uses Thrust to sort and group indices, with ATen bookkeeping around it.
- `aten_conv1d`: cuDNN (CONDITIONAL_BACKEND); `aten/src/ATen/native/Convolution.cpp`; The public convolution dispatcher can select the cuDNN backend when its guards hold.
- `aten_conv2d`: cuDNN|cuBLAS (CONDITIONAL_BACKEND); `aten/src/ATen/native/Convolution.cpp|aten/src/ATen/native/cuda/ConvolutionMM2d.cu`; Public convolution may select cuDNN; the explicit slow CUDA 2-D path is im2col plus cuBLAS GEMM.
- `aten_conv3d`: cuDNN (CONDITIONAL_BACKEND); `aten/src/ATen/native/Convolution.cpp`; The public convolution dispatcher can select the cuDNN backend when its guards hold.
- `aten_conv_tbc_backward_cpu`: cuBLAS (ATEN_COMPOSITION); `aten/src/ATen/native/ConvolutionTBC.cpp`; CompositeExplicitAutograd backward invokes addmm, whose CUDA implementation uses cuBLAS.
- `aten_conv_tbc_cpu`: cuBLAS (ATEN_COMPOSITION); `aten/src/ATen/native/ConvolutionTBC.cpp`; CompositeExplicitAutograd conv_tbc invokes addmm, whose CUDA implementation uses cuBLAS.
- `aten_conv_transpose2d`: cuBLAS (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/cuda/NaiveConvolutionTranspose2d.cu`; The explicit slow CUDA transpose-convolution path combines GEMM with ATen column kernels.
- `aten_conv_transpose3d_backward_cpu`: cuBLAS (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/cuda/NaiveConvolutionTranspose3d.cu`; The CUDA backward path combines GEMM with ATen volume-column kernels.
- `aten_conv_transpose3d_cpu`: cuBLAS (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/cuda/NaiveConvolutionTranspose3d.cu`; The CUDA counterpart combines GEMM with ATen volume-column kernels.
- `aten_conv_transpose3d_grad_weight_cpu`: cuBLAS (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/cuda/NaiveConvolutionTranspose3d.cu`; The CUDA weight-gradient path combines GEMM with ATen volume-column kernels.
- `aten_ctc_loss_backward_cpu`: cuDNN (CONDITIONAL_BACKEND); `aten/src/ATen/native/LossCTC.cpp|aten/src/ATen/native/cudnn/LossCTC.cpp`; The cuDNN CTC route jointly computes loss and gradient; the native CUDA fallback is separate.
- `aten_ctc_loss_cpu`: cuDNN (CONDITIONAL_BACKEND); `aten/src/ATen/native/LossCTC.cpp|aten/src/ATen/native/cudnn/LossCTC.cpp`; ATen selects cuDNN only for the supported CTC shapes, dtypes, and determinism constraints.
- `aten_cumprod_backward_cpu`: CUB (ATEN_COMPOSITION); `aten/src/ATen/native/ReduceOps.cpp|aten/src/ATen/native/cuda/ScanUtils.cuh`; The composite backward formula invokes CUDA cumsum/cumprod scans, which can select CUB for contiguous innermost dimensions.
- `aten_cumprod_cpu`: CUB (TEMPLATE_GENERATED_LIBRARY); `aten/src/ATen/native/cuda/ScanUtils.cuh`; The contiguous innermost-dimension scan uses CUB; other layouts use ATen scan kernels.
- `aten_cumsum`: CUB (TEMPLATE_GENERATED_LIBRARY); `aten/src/ATen/native/cuda/ScanUtils.cuh`; The contiguous innermost-dimension scan uses CUB; other layouts use ATen scan kernels.
- `aten_dilated_convolution_cpu`: cuBLAS (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/cuda/NaiveDilatedConvolution.cu`; The explicit dilated CUDA fallback uses BLAS plus ATen unfold/fold kernels.
- `aten_dot`: cuBLAS (LINKED_FIXED_LIBRARY); `aten/src/ATen/native/cuda/Blas.cpp`; Dense CUDA dot dispatches to the cuBLAS dot wrapper.
- `aten_flash_attention_backward_cpu`: CUTLASS|cuDNN (CONDITIONAL_BACKEND); `aten/src/ATen/native/transformers/cuda/attention_backward.cu|aten/src/ATen/native/transformers/cuda/attention.cu`; CUDA attention backward selects a template or cuDNN backend under runtime guards.
- `aten_flash_attention_cpu`: CUTLASS|cuDNN (CONDITIONAL_BACKEND); `aten/src/ATen/native/transformers/cuda/attention.cu`; CUDA scaled-dot-product attention selects among template flash/efficient kernels and cuDNN under runtime guards.
- `aten_hspmm_cpu`: cuSPARSE (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu|aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp`; The hybrid sparse result path prepares sparse metadata and delegates the dense product to cuSPARSE.
- `aten_index_put_impl_cpu`: CUB (CONDITIONAL_TEMPLATE_BACKEND); `aten/src/ATen/native/cuda/Indexing.cu`; The deterministic/sorted CUDA index_put path uses CUB; the ordinary atomic path does not.
- `aten_int_mm_out_cpu`: cuBLAS (LINKED_FIXED_LIBRARY); `aten/src/ATen/native/cuda/Blas.cpp`; The CUDA int8 matrix product uses the cuBLAS int8 GEMM wrapper.
- `aten_logcumsumexp_cpu`: CUB (TEMPLATE_GENERATED_LIBRARY); `aten/src/ATen/native/cuda/ScanUtils.cuh`; The scan framework can use CUB for a contiguous innermost dimension.
- `aten_masked_scatter_cpu`: CUB (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/cuda/IndexKernel.cu`; CUB computes mask offsets; ATen kernels validate and scatter values.
- `aten_masked_select_cpu`: CUB (ATEN_COMPOSITION); `aten/src/ATen/native/cuda/IndexKernel.cpp|aten/src/ATen/native/cuda/Nonzero.cu`; masked_select delegates mask indexing to the CUDA indexing/nonzero machinery.
- `aten_masked_select_serial_cpu`: CUB (NO_EXACT_CUDA_COUNTERPART); `aten/src/ATen/native/cuda/IndexKernel.cpp|aten/src/ATen/native/cuda/Nonzero.cu`; CUDA has one masked-select route rather than the CPU serial helper; that route reaches CUB-backed indexing.
- `aten_mm`: cuBLAS (LINKED_FIXED_LIBRARY); `aten/src/ATen/native/cuda/Blas.cpp`; Dense CUDA mm dispatches to GEMM.
- `aten_mode_cpu`: Thrust (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/cuda/TensorModeKernel.cu`; ATen uses Thrust sorting/reduction/search inside the CUDA mode implementation.
- `aten_mv`: cuBLAS (LINKED_FIXED_LIBRARY); `aten/src/ATen/native/cuda/Blas.cpp`; Dense CUDA mv dispatches to GEMV.
- `aten_nested_bmm_cpu`: CUTLASS (TEMPLATE_GENERATED_LIBRARY); `aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu`; Nested CUDA bmm instantiates CUTLASS grouped GEMM kernels.
- `aten_nested_matmul_broadcast_cpu`: CUTLASS (ATEN_COMPOSITION); `aten/src/ATen/native/nested/NestedTensorMatmul.cpp|aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu`; The composite nested matmul route reaches CUTLASS grouped GEMM when it lowers to nested bmm.
- `aten_nonzero_out_cpu`: CUB (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/cuda/Nonzero.cu`; ATen kernels form flags/indices around CUB selection and reduction primitives.
- `aten_randperm_cpu`: CUB (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/cuda/Randperm.cu`; ATen generates random keys and uses CUB radix sort to form the permutation.
- `aten_sampled_addmm_sparse_csr_cpu`: cuSPARSE (LINKED_FIXED_LIBRARY); `aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp`; The SparseCsrCUDA implementation delegates sampled dense-dense multiplication to cuSPARSE SDDMM.
- `aten_segment_reduce_lengths_cpu`: CUB (CONDITIONAL_TEMPLATE_BACKEND); `aten/src/ATen/native/cuda/SegmentReduce.cu`; The supported contiguous segment-reduction route uses CUB; other cases use ATen CUDA kernels.
- `aten_sort_cpu`: CUB|Thrust (CONDITIONAL_TEMPLATE_BACKEND); `aten/src/ATen/native/cuda/SortStable.cu|aten/src/ATen/native/cuda/SortImpl.cu`; The chosen CUDA sorting path depends on dtype, dimension length, stability, and build configuration.
- `aten_sparse_addmm_cpu`: cuSPARSE (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu|aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp`; COO metadata is converted/prepared by ATen before the CSR sparse-dense multiply.
- `aten_sparse_addmv_bsr_cpu`: cuSPARSE (LINKED_FIXED_LIBRARY); `aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp`; The compressed sparse CUDA addmv implementation calls cuSPARSE.
- `aten_sparse_addmv_csr_cpu`: cuSPARSE (LINKED_FIXED_LIBRARY); `aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp`; The CSR CUDA addmv implementation calls cuSPARSE SpMV.
- `aten_sparse_coo_softmax_backward_cpu`: Thrust (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/sparse/cuda/SoftMax.cu`; The CUDA sparse-softmax backward path uses Thrust primitives plus ATen kernels.
- `aten_sparse_coo_softmax_cpu`: Thrust (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/sparse/cuda/SoftMax.cu`; The CUDA sparse-softmax path uses Thrust primitives plus ATen kernels.
- `aten_sparse_csr_addmm_cpu`: cuSPARSE (LINKED_FIXED_LIBRARY); `aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp`; The CSR CUDA addmm implementation calls cuSPARSE SpMM for supported layouts.
- `aten_topk_cpu`: CUB (MIXED_EXTERNAL_AND_ATEN_KERNELS); `aten/src/ATen/native/cuda/TensorTopK.cu`; Large top-k uses CUB scan primitives inside a larger ATen selection algorithm.
- `aten_unique_bool_cpu`: CUB|Thrust (NO_EXACT_CUDA_COUNTERPART); `aten/src/ATen/native/cuda/UniqueCub.cu|aten/src/ATen/native/cuda/Unique.cu`; CUDA unique selects CUB or Thrust machinery; it has no separate CPU-bool helper counterpart.
- `aten_unique_consecutive_cpu`: CUB|Thrust (TEMPLATE_GENERATED_LIBRARY); `aten/src/ATen/native/cuda/UniqueCub.cu|aten/src/ATen/native/cuda/Unique.cu`; CUDA unique-consecutive is implemented with template-library scans/run-length operations.
- `aten_unique_dim_impl_cpu`: Thrust (TEMPLATE_GENERATED_LIBRARY); `aten/src/ATen/native/cuda/Unique.cu`; Dimension-wise CUDA unique uses Thrust sorting and uniquing.
- `aten_unique_dim_template_cpu`: Thrust (TEMPLATE_GENERATED_LIBRARY); `aten/src/ATen/native/cuda/Unique.cu`; Dimension-wise CUDA unique uses Thrust sorting and uniquing.
- `aten_unique_sorted_cpu`: CUB|Thrust (TEMPLATE_GENERATED_LIBRARY); `aten/src/ATen/native/cuda/UniqueCub.cu|aten/src/ATen/native/cuda/Unique.cu`; CUDA unique uses CUB for the common flattened path and Thrust for dimension-wise paths.
