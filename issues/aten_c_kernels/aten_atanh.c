/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float tanhf(float);
extern ATEN_CONST float atanhf(float);
void aten_atanh(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = atanhf(x[i]);
  }
#pragma endscop
}
