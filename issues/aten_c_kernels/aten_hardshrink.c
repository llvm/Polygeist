/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_hardshrink(float self[N], float lambd, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = self[i] >= -lambd && self[i] <= lambd ? 0.0f : self[i];
  }
#pragma endscop
}
