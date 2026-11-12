/* Contiguous scalar row from pinned ATen _vec_softmax_backward<float>.
 * SoftMaxKernel.cpp:208-286, BSD license in third_party/pytorch/LICENSE.
 * Specialize outer_size=inner_size=1, retaining the complete scalar tail's
 * reduction order and FP32 arithmetic. The one-element temporary buffer is
 * a local scalar. Vector paths have zero iterations for inner_size=1.
 * This equivalent reference is bound to cuDNN SoftmaxBackward; cuDNN source
 * is unavailable and its numerical behavior is validated separately. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
void NETLIB_SPEC(int n, float grad[], float output[], float result[]) {
  float sum = 0.0f;
  for (int dim_idx = 0; dim_idx < n; ++dim_idx)
    sum += grad[dim_idx] * output[dim_idx];
  for (int dim_idx = 0; dim_idx < n; ++dim_idx)
    result[dim_idx] = output[dim_idx] * (grad[dim_idx] - sum);
}
