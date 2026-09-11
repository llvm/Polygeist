/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float calc_laguerre_lf(float, float);
void aten_laguerre_polynomial_l(float a[N], float b[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = calc_laguerre_lf(a[i], b[i]);
  }
#pragma endscop
}
