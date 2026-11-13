/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_elu_backward(float grad[N], float output[N], float alpha, float scale, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = output[i] <= 0.0f ? grad[i] * (output[i] + alpha) * scale : grad[i] * scale;
  }
#pragma endscop
}
