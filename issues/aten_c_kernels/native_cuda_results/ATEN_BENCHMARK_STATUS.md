# ATen benchmark campaign status

All 116 native GPU and x86 CPU baselines use the exact recorded shape/dtype, five warmups, and best-of-20 synchronized wall time. Raised ratios are emitted only after a device-pointer output comparison against the extracted C reference.

- Native GPU baselines: 116/116
- Native x86 CPU baselines: 116/116
- Raised resident values: 78/116
- Strictly device-verified raised values: 77/116
- Legally comparable raised/native ratios: 71/116
- Median raised/native ratio (eligible rows only): 2.830x

## Raised status

- BUILD_OR_LOWERING_BLOCKED: 4
- LEGACY_RESIDENT: 1
- NO_COMPLETE_CURRENT_LIBRARY_REWRITE: 21
- RESIDUAL_IR_BLOCKED: 11
- SEMANTIC_OR_RUNTIME_FAILURE: 2
- VERIFIED_RESIDENT: 77

## Non-verified breakdown

- BUILD_OR_LOWERING_BLOCKED: aten_mean, aten_mse_loss, aten_sparse_csr_reduce_all_cpu, aten_sparse_sum_cpu
- LEGACY_RESIDENT: aten_conv2d_columns_cpu
- NO_COMPLETE_CURRENT_LIBRARY_REWRITE: aten_adaptive_avg_pool2d_backward_cpu, aten_adaptive_avg_pool3d_backward_cpu, aten_adaptive_max_pool2d_backward_cpu, aten_cat_sparse_cpu, aten_cumprod_backward_cpu, aten_dense_sparse_add_cpu, aten_fractional_max_pool2d_cpu, aten_fractional_max_pool3d_cpu, aten_host_softmax_backward_cpu, aten_host_softmax_cpu, aten_layer_norm, aten_layer_norm_backward_cpu, aten_layer_norm_cpu_backend, aten_nested_softmax_backward_cpu, aten_norm_cpu, aten_rms_norm, aten_sort_cpu, aten_sparse_coo_softmax_backward_cpu, aten_topk_cpu, aten_vector_norm_out_cpu, aten_weight_norm_backward_cpu
- RESIDUAL_IR_BLOCKED: aten_binary_cross_entropy, aten_bmm, aten_conv3d, aten_count_nonzero_cpu, aten_count_nonzero_impl_cpu, aten_dot, aten_im2col, aten_mm, aten_mv, aten_softmax, aten_split_copy_cpu
- SEMANTIC_OR_RUNTIME_FAILURE: aten_cumsum, aten_div_floor

## Missing raised resident values

aten_adaptive_avg_pool2d_backward_cpu, aten_adaptive_avg_pool3d_backward_cpu, aten_adaptive_max_pool2d_backward_cpu, aten_binary_cross_entropy, aten_bmm, aten_cat_sparse_cpu, aten_conv3d, aten_count_nonzero_cpu, aten_count_nonzero_impl_cpu, aten_cumprod_backward_cpu, aten_cumsum, aten_dense_sparse_add_cpu, aten_div_floor, aten_dot, aten_fractional_max_pool2d_cpu, aten_fractional_max_pool3d_cpu, aten_host_softmax_backward_cpu, aten_host_softmax_cpu, aten_im2col, aten_layer_norm, aten_layer_norm_backward_cpu, aten_layer_norm_cpu_backend, aten_mean, aten_mm, aten_mse_loss, aten_mv, aten_nested_softmax_backward_cpu, aten_norm_cpu, aten_rms_norm, aten_softmax, aten_sort_cpu, aten_sparse_coo_softmax_backward_cpu, aten_sparse_csr_reduce_all_cpu, aten_sparse_sum_cpu, aten_split_copy_cpu, aten_topk_cpu, aten_vector_norm_out_cpu, aten_weight_norm_backward_cpu

The CSV is the authoritative per-kernel artifact. Legacy resident values are
withheld from the HTML and are not used for paper ratios until rerun through
the strict device-output gate. Historical mapped-host measurements are ABI
diagnostics only; they do not feed the ATen results or slowness pages.
