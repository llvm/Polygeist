/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_clamp_scalar_cpu(float x[N], float minval, float maxval, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = x[i] < minval ? minval : (x[i] > maxval ? maxval : x[i]);
  }
#pragma endscop
}
