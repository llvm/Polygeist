/* Scalar FP32 contiguous specialization of pinned ATen addcmul_cpu_kernel.
 * TensorIterator mapping becomes a C loop; scalar operands remain parameters.
 * Scalar body and operation order are retained. See manifest provenance. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
void NETLIB_SPEC(int n, float self_val[], float t1_val[], float t2_val[], float output[], float scalar_val) {
  for (int i=0;i<n;++i) {
    output[i] = self_val[i] + scalar_val * t1_val[i] * t2_val[i];
  }
}
