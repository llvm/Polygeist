/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float tanhf(float);
void aten_gelu_cpu_tanh(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float inner = 0.7978845608028654f * (x[i] + 0.044715f * x[i] * x[i] * x[i]);
    out[i] = 0.5f * x[i] * (1.0f + tanhf(inner));
  }
#pragma endscop
}
