/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_hardsigmoid_backward(float grad[N], float self[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = self[i] > -3.0f && self[i] < 3.0f ? grad[i] * (1.0f / 6.0f) : 0.0f;
  }
#pragma endscop
}
