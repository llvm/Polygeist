/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float logf(float);
void aten_logit(float x[N], float eps, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float z = x[i] < eps ? eps : (x[i] > 1.0f - eps ? 1.0f - eps : x[i]);
    out[i] = logf(z / (1.0f - z));
  }
#pragma endscop
}
