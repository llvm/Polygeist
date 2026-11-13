/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float calc_scaled_bessel_k1f(float);
void aten_scaled_modified_bessel_k1(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = calc_scaled_bessel_k1f(x[i]);
  }
#pragma endscop
}
