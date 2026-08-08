/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float tanhf(float);
void aten_gelu_backward_cpu_tanh(float grad[N], float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float x2 = x[i] * x[i];
    float inner = 0.7978845608028654f * (x[i] + 0.044715f * x[i] * x2);
    float t = tanhf(inner);
    float deriv = 0.5f * (1.0f + t) + 0.5f * x[i] * (1.0f - t * t) * 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * x2);
    out[i] = grad[i] * deriv;
  }
#pragma endscop
}
