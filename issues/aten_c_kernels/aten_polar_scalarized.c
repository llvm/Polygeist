/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float cosf(float);
extern ATEN_CONST float sinf(float);
void aten_polar_scalarized(float magnitude[N], float angle[N], float out_re[N], float out_im[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out_re[i] = magnitude[i] * cosf(angle[i]);
    out_im[i] = magnitude[i] * sinf(angle[i]);
  }
#pragma endscop
}
