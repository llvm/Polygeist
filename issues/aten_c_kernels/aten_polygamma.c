/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float calc_polygammaf(int, float);
void aten_polygamma(float x[N], int order, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = calc_polygammaf(order, x[i]);
  }
#pragma endscop
}
