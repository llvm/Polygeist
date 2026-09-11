/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_smooth_l1_elementwise(float a[N], float b[N], float beta, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float z = a[i] - b[i];
    float az = z < 0.0f ? -z : z;
    out[i] = az < beta ? 0.5f * z * z / beta : az - 0.5f * beta;
  }
#pragma endscop
}
