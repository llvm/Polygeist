/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float powf(float, float);
extern ATEN_CONST float powf(float, float);
void aten_pow(float a[N], float b[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = powf(a[i], b[i]);
  }
#pragma endscop
}
