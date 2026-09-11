/* aten::layer_norm for one row.
 * Upstream family: aten/src/ATen/native/layer_norm.cpp.
 */
#define N 128
extern float sqrtf(float);

void aten_layer_norm(float x[N], float weight[N], float bias[N],
                     float out[N], float eps) {
  float mean = 0.0f;
  float variance = 0.0f;
#pragma scop
  for (int i = 0; i < N; ++i)
    mean += x[i];
  mean /= (float)N;
  for (int i = 0; i < N; ++i) {
    float centered = x[i] - mean;
    variance += centered * centered;
  }
  float inv_std = 1.0f / sqrtf(variance / (float)N + eps);
  for (int i = 0; i < N; ++i)
    out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
#pragma endscop
}
