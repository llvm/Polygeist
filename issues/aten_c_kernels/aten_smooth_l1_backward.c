/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_smooth_l1_backward(float input[N], float target[N], float norm, float beta, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float z = input[i] - target[i];
    out[i] = z <= -beta ? -norm : (z >= beta ? norm : norm * z / beta);
  }
#pragma endscop
}
