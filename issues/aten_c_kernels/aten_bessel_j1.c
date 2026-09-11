/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float bessel_j1_forwardf(float);
void aten_bessel_j1(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = bessel_j1_forwardf(x[i]);
  }
#pragma endscop
}
