/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float hypotf(float, float);
extern ATEN_CONST float hypotf(float, float);
void aten_hypot(float a[N], float b[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = hypotf(a[i], b[i]);
  }
#pragma endscop
}
