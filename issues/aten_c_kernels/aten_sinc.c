/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float sinf(float);
void aten_sinc(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = x[i] == 0.0f ? 1.0f : sinf(3.14159265358979323846f * x[i]) / (3.14159265358979323846f * x[i]);
  }
#pragma endscop
}
