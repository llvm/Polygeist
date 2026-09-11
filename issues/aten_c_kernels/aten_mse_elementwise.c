/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_mse_elementwise(float a[N], float b[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = (a[i] - b[i]) * (a[i] - b[i]);
  }
#pragma endscop
}
