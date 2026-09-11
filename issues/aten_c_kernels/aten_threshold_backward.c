/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_threshold_backward(float grad[N], float self[N], float threshold, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = self[i] <= threshold ? 0.0f : grad[i];
  }
#pragma endscop
}
