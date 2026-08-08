/* aten::rms_norm for one row, with learned elementwise weight. */
#ifndef N
#define N 128
#endif
extern float sqrtf(float);

void aten_rms_norm(float x[N], float weight[N], float out[N], float eps) {
  float sum_square = 0.0f;
#pragma scop
  for (int i = 0; i < N; ++i)
    sum_square += x[i] * x[i];
  float scale = 1.0f / sqrtf(sum_square / (float)N + eps);
  for (int i = 0; i < N; ++i)
    out[i] = weight[i] * (scale * x[i]);
#pragma endscop
}
