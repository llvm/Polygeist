/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float expf(float);
void aten_glu(float a[N], float b[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = a[i] / (1.0f + expf(-b[i]));
  }
#pragma endscop
}
