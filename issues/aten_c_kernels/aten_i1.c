/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float calc_i1f(float);
void aten_i1(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = calc_i1f(x[i]);
  }
#pragma endscop
}
