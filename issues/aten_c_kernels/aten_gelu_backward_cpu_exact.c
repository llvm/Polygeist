/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float erff(float);
extern ATEN_CONST float expf(float);
void aten_gelu_backward_cpu_exact(float grad[N], float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float cdf = 0.5f * (1.0f + erff(x[i] * 0.7071067811865475f));
    float pdf = 0.3989422804014327f * expf(-0.5f * x[i] * x[i]);
    out[i] = grad[i] * (cdf + x[i] * pdf);
  }
#pragma endscop
}
