/* FP32 contiguous specialization of pinned ATen scalar code.
 * TensorIterator/vector mapping becomes a loop; float math overloads
 * use equivalent C functions. See the manifest for provenance. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
void NETLIB_SPEC(int n, float self[], float other[], float output[], float alpha) {
  for (int i = 0; i < n; ++i) {
    output[i] = self[i] + alpha * other[i];
  }
}
