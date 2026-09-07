# ATen benchmark campaign status

The resolved campaign contains 242 shape/dtype specifications. Available native GPU and x86 CPU baselines use the exact recorded shape/dtype, five warmups, and best-of-20 synchronized wall time. Raised ratios are emitted only after a device-pointer output comparison against the extracted C reference.

- Native GPU baselines: 242/242
- Native x86 CPU baselines: 116/242
- Raised resident values: 178/242
- Strictly device-verified raised values: 98/242
- Legally comparable raised/native ratios: 71/242
- Median raised/native ratio (eligible rows only): 2.830x

## Raised status

- BUILD_OR_LOWERING_BLOCKED: 4
- LEGACY_RESIDENT: 80
- NO_COMPLETE_CURRENT_LIBRARY_REWRITE: 38
- RESIDUAL_IR_BLOCKED: 22
- VERIFIED_RESIDENT: 98

## Non-verified breakdown

- BUILD_OR_LOWERING_BLOCKED: aten_aminmax_allreduce_cpu, aten_conv_transpose3d_backward_cpu, aten_linear_combination_cpu, aten_renorm_scale_factor
- LEGACY_RESIDENT: aten_addcdiv, aten_addcmul, aten_angle_complex_scalarized, aten_angle_real, aten_as_complex_cpu, aten_atan2, aten_batch_norm_cpu_entry, aten_clamp, aten_clamp_cpu, aten_clamp_max_scalar_cpu, aten_clamp_min_scalar_cpu, aten_clamp_scalar_cpu, aten_complex_scalarized, aten_conj_complex_scalarized, aten_conv1d, aten_conv2d_columns_cpu, aten_copy_cpu, aten_copy_tensor_array_cpu, aten_cross, aten_cross_cpu_backend, aten_dilated_convolution_cpu, aten_expm1, aten_fmax, aten_fmin, aten_fmod, aten_gelu_backward_cpu_exact, aten_gelu_backward_cpu_tanh, aten_gelu_cpu_tanh, aten_glu, aten_glu_backward, aten_glu_jvp, aten_hardshrink, aten_hardsigmoid, aten_hardsigmoid_backward, aten_hardtanh, aten_hardtanh_backward, aten_huber_backward, aten_huber_elementwise, aten_int_mm_out_cpu, aten_lerp, aten_lerp_scalar, aten_lerp_scalar_cpu, aten_lerp_tensor_cpu, aten_log10, aten_log1p, aten_log2, aten_log_normal_cpu, aten_log_sigmoid_backward_cpu, aten_logit, aten_logit_backward, aten_masked_scale, aten_maximum, aten_minimum, aten_mse_backward, aten_mse_elementwise, aten_narrow_copy_dense_cpu, aten_nested_clone_cpu, aten_nested_matmul_broadcast_cpu, aten_nested_squeeze_cpu, aten_normal_cpu, aten_pixel_shuffle, aten_pixel_shuffle_cpu_backend, aten_pixel_unshuffle_cpu_backend, aten_polar_scalarized, aten_round, aten_round_decimals, aten_shrink_backward, aten_smooth_l1_backward, aten_smooth_l1_elementwise, aten_softshrink, aten_sparse_intersection_apply_cpu, aten_sparse_intersection_launch_cpu, aten_stack_serial_cpu, aten_threshold_backward, aten_transpose_copy, aten_trunc, aten_unbind_copy_cpu, aten_uniform_cpu, aten_upsample_bilinear2d, aten_zeros_cpu
- NO_COMPLETE_CURRENT_LIBRARY_REWRITE: aten_adaptive_avg_pool2d_backward_cpu, aten_adaptive_avg_pool3d_backward_cpu, aten_adaptive_max_pool2d_backward_cpu, aten_binary_cross_entropy, aten_bmm, aten_cat_sparse_cpu, aten_conv3d, aten_count_nonzero_cpu, aten_count_nonzero_impl_cpu, aten_cumprod_backward_cpu, aten_cumsum, aten_dense_sparse_add_cpu, aten_div_floor, aten_dot, aten_fractional_max_pool2d_cpu, aten_fractional_max_pool3d_cpu, aten_host_softmax_backward_cpu, aten_host_softmax_cpu, aten_im2col, aten_layer_norm, aten_layer_norm_backward_cpu, aten_layer_norm_cpu_backend, aten_mean, aten_mm, aten_mse_loss, aten_mv, aten_nested_softmax_backward_cpu, aten_norm_cpu, aten_rms_norm, aten_softmax, aten_sort_cpu, aten_sparse_coo_softmax_backward_cpu, aten_sparse_csr_reduce_all_cpu, aten_sparse_sum_cpu, aten_split_copy_cpu, aten_topk_cpu, aten_vector_norm_out_cpu, aten_weight_norm_backward_cpu
- RESIDUAL_IR_BLOCKED: aten_addr_elementwise, aten_aminmax_cpu, aten_bilinear_cpu, aten_blas_sum_cpu, aten_block_diag_cpu, aten_div_trunc, aten_equal_cpu, aten_gradient_cpu, aten_log_sigmoid_cpu, aten_max_all_cpu, aten_max_reduce_cpu, aten_max_values_cpu, aten_min_all_cpu, aten_min_reduce_cpu, aten_min_values_cpu, aten_or_reduce_cpu, aten_prod, aten_sum_cpu_backend, aten_trace_cpu, aten_trilinear_cpu, aten_unfolded2d_copy_cpu, aten_xor_sum_cpu

## Missing raised resident values

aten_adaptive_avg_pool2d_backward_cpu, aten_adaptive_avg_pool3d_backward_cpu, aten_adaptive_max_pool2d_backward_cpu, aten_addr_elementwise, aten_aminmax_allreduce_cpu, aten_aminmax_cpu, aten_bilinear_cpu, aten_binary_cross_entropy, aten_blas_sum_cpu, aten_block_diag_cpu, aten_bmm, aten_cat_sparse_cpu, aten_conv3d, aten_conv_transpose3d_backward_cpu, aten_count_nonzero_cpu, aten_count_nonzero_impl_cpu, aten_cumprod_backward_cpu, aten_cumsum, aten_dense_sparse_add_cpu, aten_div_floor, aten_div_trunc, aten_dot, aten_equal_cpu, aten_fractional_max_pool2d_cpu, aten_fractional_max_pool3d_cpu, aten_gradient_cpu, aten_host_softmax_backward_cpu, aten_host_softmax_cpu, aten_im2col, aten_layer_norm, aten_layer_norm_backward_cpu, aten_layer_norm_cpu_backend, aten_linear_combination_cpu, aten_log_sigmoid_cpu, aten_max_all_cpu, aten_max_reduce_cpu, aten_max_values_cpu, aten_mean, aten_min_all_cpu, aten_min_reduce_cpu, aten_min_values_cpu, aten_mm, aten_mse_loss, aten_mv, aten_nested_softmax_backward_cpu, aten_norm_cpu, aten_or_reduce_cpu, aten_prod, aten_renorm_scale_factor, aten_rms_norm, aten_softmax, aten_sort_cpu, aten_sparse_coo_softmax_backward_cpu, aten_sparse_csr_reduce_all_cpu, aten_sparse_sum_cpu, aten_split_copy_cpu, aten_sum_cpu_backend, aten_topk_cpu, aten_trace_cpu, aten_trilinear_cpu, aten_unfolded2d_copy_cpu, aten_vector_norm_out_cpu, aten_weight_norm_backward_cpu, aten_xor_sum_cpu

The CSV is the authoritative per-kernel artifact. Legacy resident values are shown for completeness but are not used for paper ratios until rerun through the strict device-output gate.
