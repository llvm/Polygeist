/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_isposinf(float x[N], float max_finite, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = (float)(x[i] > max_finite);
  }
#pragma endscop
}
