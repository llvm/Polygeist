/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float erff(float);
void aten_gelu_cpu_exact(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.7071067811865475f));
  }
#pragma endscop
}
