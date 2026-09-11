/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float calc_erfcxf(float);
void aten_erfcx(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = calc_erfcxf(x[i]);
  }
#pragma endscop
}
