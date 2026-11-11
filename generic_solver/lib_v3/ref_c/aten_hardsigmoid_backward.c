/* Scalar FP32 contiguous specialization of pinned ATen hardsigmoid_backward_kernel.
 * TensorIterator mapping becomes a C loop; scalar operands remain parameters.
 * Scalar body and operation order are retained. See manifest provenance. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
void NETLIB_SPEC(int n, float grad_val[], float self_val[], float output[]) {
  const float zero = 0.0f;
  const float three = 3.0f;
  const float neg_three = -3.0f;
  const float one_sixth = 1.0f / 6.0f;
  for (int i=0;i<n;++i) {
    output[i] = (self_val[i] > neg_three && self_val[i] < three)
    ? grad_val[i] * one_sixth
    : zero;
  }
}
