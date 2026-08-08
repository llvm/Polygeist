/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float roundf(float);
void aten_round_decimals(float x[N], float scale, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = roundf(x[i] * scale) / scale;
  }
#pragma endscop
}
