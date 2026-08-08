/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float logf(float);
void aten_entr(float x[N], float nan_value, float neg_inf, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = x[i] < 0.0f ? nan_value : (x[i] == 0.0f ? 0.0f : (x[i] <= 1.0f ? -x[i] * logf(x[i]) : neg_inf));
  }
#pragma endscop
}
