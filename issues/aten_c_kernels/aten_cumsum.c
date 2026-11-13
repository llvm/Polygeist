/* aten::cumsum over a vector. Upstream: ATen/native/cpu/ReduceOpsKernel.cpp. */
#ifndef N
#define N 256
#endif
void aten_cumsum(float input[N], float output[N]) {
  float sum = 0.0f;
#pragma scop
  for (int i = 0; i < N; ++i) {
    sum += input[i];
    output[i] = sum;
  }
#pragma endscop
}
