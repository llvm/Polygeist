/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_log_sigmoid_backward_cpu(float input[N], float buffer[N], float grad[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float neg = input[i] < 0.0f;
    float max_deriv = neg ? 1.0f : 0.0f;
    float sign = neg ? 1.0f : -1.0f;
    out[i] = (max_deriv - sign * (buffer[i] / (1.0f + buffer[i]))) * grad[i];
  }
#pragma endscop
}
