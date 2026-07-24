/* aten::prod over a vector. Upstream: ATen/native/cpu/ReduceOpsKernel.cpp. */
#define N 256
void aten_prod(float input[N], float out[1]) {
#pragma scop
  out[0] = 1.0f;
  for (int i = 0; i < N; ++i) out[0] *= input[i];
#pragma endscop
}
