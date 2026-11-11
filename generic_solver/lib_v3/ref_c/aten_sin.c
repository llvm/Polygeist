/* FP32 contiguous specialization of pinned ATen scalar semantics.
 * TensorIterator/vector map is expanded as a loop; std math overloads
 * become equivalent float C functions. errno is not observed.
 * The manifest pins the upstream sources. No fast-math is assumed.
 */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
extern __attribute__((const)) float sinf(float);
void NETLIB_SPEC(int n, float x[], float output[]) {
  for (int i = 0; i < n; ++i) { output[i] = sinf(x[i]); }
}
