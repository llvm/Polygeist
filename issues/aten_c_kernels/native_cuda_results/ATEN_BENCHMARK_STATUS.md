# ATen benchmark campaign status

The resolved campaign contains 292 shape/dtype specifications. Available native GPU, x86 CPU, and Jetson CPU measurements use the exact recorded shape/dtype, five warmups, and best-of-20 synchronized wall time. The Jetson CPU framework version is recorded separately in provenance. Raised ratios are emitted only after a device-pointer output comparison against the extracted C reference.

- Native GPU baselines: 242/292
- Native x86 CPU baselines: 116/292
- Native Jetson CPU baselines: 243/292
- Raised resident values: 216/292
- Strictly device-verified raised values: 215/292
- Legally comparable raised/native ratios: 113/292
- Median raised/native ratio (eligible rows only): 2.742x

## Raised status

- BUILD_OR_LOWERING_BLOCKED: 5
- NO_COMPLETE_CURRENT_LIBRARY_REWRITE: 13
- RESIDUAL_IR_BLOCKED: 14
- SEMANTIC_OR_RUNTIME_FAILURE: 45
- VERIFIED_RESIDENT: 215

## Non-verified breakdown

- BUILD_OR_LOWERING_BLOCKED: aten_cartesian_prod_cpu, aten_flip_tensor_transform_cpu, aten_layer_norm, aten_mse_loss, aten_rms_norm
- NO_COMPLETE_CURRENT_LIBRARY_REWRITE: aten_cat_sparse_cpu, aten_cumprod_backward_cpu, aten_dense_sparse_add_cpu, aten_fractional_max_pool2d_cpu, aten_fractional_max_pool3d_cpu, aten_host_softmax_cpu, aten_layer_norm_backward_cpu, aten_layer_norm_cpu_backend, aten_norm_cpu, aten_sort_cpu, aten_topk_cpu, aten_vector_norm_out_cpu, aten_weight_norm_backward_cpu
- RESIDUAL_IR_BLOCKED: aten_adaptive_avg_pool2d_backward_cpu, aten_adaptive_avg_pool3d_backward_cpu, aten_binary_cross_entropy, aten_blas_sum_cpu, aten_conv3d, aten_max_all_cpu, aten_mean, aten_min_all_cpu, aten_mv, aten_nested_batch_offsets_cpu, aten_prod, aten_sparse_csr_reduce_all_cpu, aten_sparse_sum_cpu, aten_trace_cpu
- SEMANTIC_OR_RUNTIME_FAILURE: aten_adaptive_max_pool2d_backward_cpu, aten_addr_elementwise, aten_aminmax_allreduce_cpu, aten_aminmax_cpu, aten_bf16_dot_cpu, aten_bf16_gemv_trans_cpu, aten_bilinear_cpu, aten_blas_dot_naive_cpu, aten_blas_gemv_generic_cpu, aten_block_diag_cpu, aten_conv_transpose3d_backward_cpu, aten_count_nonzero_cpu, aten_cumsum, aten_div_floor, aten_div_trunc, aten_dot, aten_equal_cpu, aten_fast_cat_dim0_cpu, aten_fp16_dot_cpu, aten_fp16_gemv_f16arith_cpu, aten_fp16_gemv_f32arith_cpu, aten_fp16_gemv_notrans_cpu, aten_fp16_gemv_trans_cpu, aten_gemm_notrans_cpu, aten_gemm_transa_cpu, aten_gemm_transab_cpu, aten_gemm_transb_cpu, aten_gradient_cpu, aten_gradient_float_cpu, aten_histogram_select_outer_bin_edges_cpu, aten_host_softmax_backward_cpu, aten_linear_combination_cpu, aten_max_reduce_cpu, aten_max_values_cpu, aten_min_reduce_cpu, aten_min_values_cpu, aten_nested_softmax_backward_cpu, aten_slow_conv3d_forward_cpu, aten_softmax, aten_sparse_coo_softmax_backward_cpu, aten_split_copy_cpu, aten_sum_cpu_backend, aten_trilinear_cpu, aten_unfolded2d_copy_cpu, aten_upsample_bilinear2d

## Missing raised resident values

aten_adaptive_avg_pool2d_backward_cpu, aten_adaptive_avg_pool3d_backward_cpu, aten_adaptive_max_pool2d_backward_cpu, aten_addr_elementwise, aten_aminmax_allreduce_cpu, aten_aminmax_cpu, aten_bf16_dot_cpu, aten_bf16_gemv_trans_cpu, aten_bilinear_cpu, aten_binary_cross_entropy, aten_blas_dot_naive_cpu, aten_blas_gemv_generic_cpu, aten_blas_sum_cpu, aten_block_diag_cpu, aten_cartesian_prod_cpu, aten_cat_sparse_cpu, aten_conv3d, aten_conv_transpose3d_backward_cpu, aten_count_nonzero_cpu, aten_cumprod_backward_cpu, aten_cumsum, aten_dense_sparse_add_cpu, aten_div_floor, aten_div_trunc, aten_dot, aten_equal_cpu, aten_fast_cat_dim0_cpu, aten_flip_tensor_transform_cpu, aten_fp16_dot_cpu, aten_fp16_gemv_f16arith_cpu, aten_fp16_gemv_f32arith_cpu, aten_fp16_gemv_notrans_cpu, aten_fp16_gemv_trans_cpu, aten_fractional_max_pool2d_cpu, aten_fractional_max_pool3d_cpu, aten_gemm_notrans_cpu, aten_gemm_transa_cpu, aten_gemm_transab_cpu, aten_gemm_transb_cpu, aten_gradient_cpu, aten_gradient_float_cpu, aten_histogram_select_outer_bin_edges_cpu, aten_host_softmax_backward_cpu, aten_host_softmax_cpu, aten_layer_norm, aten_layer_norm_backward_cpu, aten_layer_norm_cpu_backend, aten_linear_combination_cpu, aten_max_all_cpu, aten_max_reduce_cpu, aten_max_values_cpu, aten_mean, aten_min_all_cpu, aten_min_reduce_cpu, aten_min_values_cpu, aten_mse_loss, aten_mv, aten_nested_batch_offsets_cpu, aten_nested_softmax_backward_cpu, aten_norm_cpu, aten_prod, aten_rms_norm, aten_slow_conv3d_forward_cpu, aten_softmax, aten_sort_cpu, aten_sparse_coo_softmax_backward_cpu, aten_sparse_csr_reduce_all_cpu, aten_sparse_sum_cpu, aten_split_copy_cpu, aten_sum_cpu_backend, aten_topk_cpu, aten_trace_cpu, aten_trilinear_cpu, aten_unfolded2d_copy_cpu, aten_vector_norm_out_cpu, aten_weight_norm_backward_cpu

The CSV is the authoritative per-kernel artifact. Legacy resident values are shown for completeness but are not used for paper ratios until rerun through the strict device-output gate.
