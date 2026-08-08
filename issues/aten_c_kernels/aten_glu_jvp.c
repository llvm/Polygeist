/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float expf(float);
void aten_glu_jvp(float result[N], float b[N], float da[N], float db[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float s = 1.0f / (1.0f + expf(-b[i]));
    out[i] = da[i] * s + result[i] * (db[i] - s * db[i]);
  }
#pragma endscop
}
