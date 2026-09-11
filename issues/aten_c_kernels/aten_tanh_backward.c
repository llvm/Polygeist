/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_tanh_backward(float grad[N], float output[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = grad[i] * (1.0f - output[i] * output[i]);
  }
#pragma endscop
}
