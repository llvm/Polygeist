/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float atan2f(float, float);
void aten_angle_complex_scalarized(float re[N], float im[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = atan2f(im[i], re[i]);
  }
#pragma endscop
}
