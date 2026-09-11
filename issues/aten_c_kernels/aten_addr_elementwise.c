/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_addr_elementwise(float self[N], float x[N], float y[N], float beta, float alpha, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = beta == 0.0f ? alpha * x[i] * y[i] : beta * self[i] + alpha * x[i] * y[i];
  }
#pragma endscop
}
