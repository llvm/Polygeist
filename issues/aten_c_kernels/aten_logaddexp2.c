/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float exp2f(float);
extern ATEN_CONST float log1pf(float);
void aten_logaddexp2(float a[N], float b[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float m = a[i] > b[i] ? a[i] : b[i];
    float d = a[i] - b[i];
    if (d < 0.0f) d = -d;
    out[i] = m + log1pf(exp2f(-d)) * 1.4426950408889634f;
  }
#pragma endscop
}
