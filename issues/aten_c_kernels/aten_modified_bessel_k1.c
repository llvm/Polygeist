/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float modified_bessel_k1_forwardf(float);
void aten_modified_bessel_k1(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = modified_bessel_k1_forwardf(x[i]);
  }
#pragma endscop
}
