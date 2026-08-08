/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float calc_chebyshev_tf(float, float);
void aten_chebyshev_polynomial_t(float a[N], float b[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = calc_chebyshev_tf(a[i], b[i]);
  }
#pragma endscop
}
