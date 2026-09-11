/* aten::leaky_relu. Upstream: ATen/native/cpu/Activation.cpp. */
#define N 256
void aten_leaky_relu(float x[N], float out[N], float slope) {
#pragma scop
  for (int i = 0; i < N; ++i) out[i] = x[i] >= 0.0f ? x[i] : slope * x[i];
#pragma endscop
}
