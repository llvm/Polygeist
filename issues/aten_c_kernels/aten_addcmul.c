/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_addcmul(float self[N], float x[N], float y[N], float value, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = self[i] + value * x[i] * y[i];
  }
#pragma endscop
}
