/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float expf(float);
extern ATEN_CONST float log1pf(float);
extern ATEN_CONST float tanhf(float);
void aten_mish(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = x[i] * tanhf(log1pf(expf(x[i])));
  }
#pragma endscop
}
