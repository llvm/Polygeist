/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_huber_backward(float input[N], float target[N], float norm, float delta, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float z = input[i] - target[i];
    out[i] = z < -delta ? -norm * delta : (z > delta ? norm * delta : norm * z);
  }
#pragma endscop
}
