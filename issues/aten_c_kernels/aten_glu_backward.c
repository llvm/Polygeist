/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_glu_backward(float sigmoid_b[N], float grad[N], float a[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = (1.0f - sigmoid_b[i]) * sigmoid_b[i] * grad[i] * a[i];
  }
#pragma endscop
}
