/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_logit_backward(float grad[N], float self[N], float eps, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = self[i] < eps || self[i] > 1.0f - eps ? 0.0f : grad[i] / (self[i] * (1.0f - self[i]));
  }
#pragma endscop
}
