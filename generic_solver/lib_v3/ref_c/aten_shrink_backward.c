/* Scalar FP32 contiguous specialization of pinned ATen shrink_backward_kernel.
 * TensorIterator mapping becomes a C loop; scalar operands remain parameters.
 * Scalar body and operation order are retained. See manifest provenance. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
void NETLIB_SPEC(int n, float grad_val[], float self_val[], float output[], float lambd_val) {
  for (int i=0;i<n;++i) {
    output[i] = (self_val[i] >= -lambd_val && self_val[i] <= lambd_val) ? 0.0f
    : grad_val[i];
  }
}
