/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float expf(float);
void aten_softplus_backward(float grad[N], float self[N], float beta, float threshold, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float z = beta * self[i];
    out[i] = z > threshold ? grad[i] : grad[i] * (1.0f - 1.0f / (1.0f + expf(z)));
  }
#pragma endscop
}
