/* aten::sigmoid. Upstream: ATen/native/cpu/UnaryOpsKernel.cpp. */
#define N 256
extern float expf(float);
void aten_sigmoid(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) out[i] = 1.0f / (1.0f + expf(-x[i]));
#pragma endscop
}
