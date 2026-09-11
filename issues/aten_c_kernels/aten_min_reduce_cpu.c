#ifndef N
#define N 4096
#endif
void aten_min_reduce_cpu(float x[N], float out[1]) {
#pragma scop
  float value = x[0];
  for (int i = 1; i < N; ++i) value = x[i] < value ? x[i] : value;
  out[0] = value;
#pragma endscop
}
