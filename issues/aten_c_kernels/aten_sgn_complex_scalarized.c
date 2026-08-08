/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float hypotf(float, float);
void aten_sgn_complex_scalarized(float re[N], float im[N], float out_re[N], float out_im[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float mag = hypotf(re[i], im[i]);
    out_re[i] = mag == 0.0f ? 0.0f : re[i] / mag;
    out_im[i] = mag == 0.0f ? 0.0f : im[i] / mag;
  }
#pragma endscop
}
