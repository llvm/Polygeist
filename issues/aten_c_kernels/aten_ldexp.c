/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float expf(float);
extern ATEN_CONST float ldexpf(float, int);
void aten_ldexp(float a[N], int exponent[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = ldexpf(a[i], exponent[i]);
  }
#pragma endscop
}
