/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_masked_scale(float x[N], float inv_scale, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = x[i] * inv_scale;
  }
#pragma endscop
}
