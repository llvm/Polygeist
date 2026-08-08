/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_mse_backward(float input[N], float target[N], float value, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = value * (input[i] - target[i]);
  }
#pragma endscop
}
