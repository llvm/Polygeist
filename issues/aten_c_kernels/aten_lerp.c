/* aten::lerp.Tensor. Upstream: ATen/native/cpu/LerpKernel.cpp. */
#define N 256
void aten_lerp(float a[N], float b[N], float weight[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) out[i] = a[i] + weight[i] * (b[i] - a[i]);
#pragma endscop
}
