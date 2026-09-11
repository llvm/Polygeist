/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float calc_kaiserf(float, float);
void aten_kaiser_window(float x[N], float beta, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = calc_kaiserf(x[i], beta);
  }
#pragma endscop
}
