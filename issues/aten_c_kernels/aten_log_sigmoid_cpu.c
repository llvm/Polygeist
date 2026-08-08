/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
extern ATEN_CONST float expf(float);
extern ATEN_CONST float log1pf(float);
void aten_log_sigmoid_cpu(float x[N], float out[N], float buffer[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float ax = x[i] < 0.0f ? -x[i] : x[i];
    buffer[i] = expf(-ax);
    out[i] = (x[i] < 0.0f ? x[i] : 0.0f) - log1pf(buffer[i]);
  }
#pragma endscop
}
