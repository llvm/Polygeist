/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_lerp_scalar(float self[N], float end[N], float weight, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = self[i] + weight * (end[i] - self[i]);
  }
#pragma endscop
}
