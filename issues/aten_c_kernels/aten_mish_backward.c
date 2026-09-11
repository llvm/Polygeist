/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float expf(float);
extern ATEN_CONST float log1pf(float);
extern ATEN_CONST float tanhf(float);
void aten_mish_backward(float grad[N], float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float s = 1.0f / (1.0f + expf(-x[i]));
    float t = tanhf(log1pf(expf(x[i])));
    out[i] = grad[i] * (t + x[i] * s * (1.0f - t * t));
  }
#pragma endscop
}
