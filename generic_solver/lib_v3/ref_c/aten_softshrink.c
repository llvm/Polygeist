/* Scalar FP32 contiguous specialization of pinned ATen softshrink_kernel.
 * TensorIterator mapping becomes a C loop; scalar operands remain parameters.
 * Scalar body and operation order are retained. See manifest provenance. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
void NETLIB_SPEC(int n, float a[], float output[], float lambd_val) {
  for (int i=0;i<n;++i) {
    output[i] = a[i] > lambd_val ? a[i] - lambd_val : (a[i] < -lambd_val ? a[i] + lambd_val : a[i] * 0.0f);
  }
}
