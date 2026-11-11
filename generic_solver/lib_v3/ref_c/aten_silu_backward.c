/* FP32 contiguous specialization of pinned ATen scalar code.
 * TensorIterator/vector mapping becomes a loop; float math overloads
 * use equivalent C functions. See the manifest for provenance. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
extern __attribute__((const)) float expf(float);
void NETLIB_SPEC(int n, float dy[], float x[], float output[]) {
  for (int i = 0; i < n; ++i) {
    const float sigmoid =
                      1.0f / (1.0f + expf(-x[i]));
                  output[i] = dy[i] * sigmoid * (1.0f + x[i] * (1.0f - sigmoid));
  }
}
