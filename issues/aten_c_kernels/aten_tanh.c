/* aten::tanh. Upstream: ATen/native/cpu/UnaryOpsKernel.cpp. */
#define N 256
extern float tanhf(float);
void aten_tanh(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) out[i] = tanhf(x[i]);
#pragma endscop
}
