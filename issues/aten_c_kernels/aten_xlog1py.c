/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float log1pf(float);
void aten_xlog1py(float x[N], float y[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = y[i] != y[i] ? y[i] : (x[i] == 0.0f ? 0.0f : x[i] * log1pf(y[i]));
  }
#pragma endscop
}
