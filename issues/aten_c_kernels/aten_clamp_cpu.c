/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_clamp_cpu(float x[N], float minval[N], float maxval[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = x[i] < minval[i] ? minval[i] : (x[i] > maxval[i] ? maxval[i] : x[i]);
  }
#pragma endscop
}
