/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_linear_combination_cpu(float input[4][N], float coefficients[4], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float value = 0.0f;
    for (int j = 0; j < 4; ++j) value += coefficients[j] * input[j][i];
    out[i] = value;
  }
#pragma endscop
}
