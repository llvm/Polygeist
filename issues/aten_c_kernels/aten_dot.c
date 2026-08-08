/* aten::dot: out[0] = sum_i x[i]*y[i]. */
#ifndef N
#define N 128
#endif

void aten_dot(double x[N], double y[N], double out[1]) {
#pragma scop
  out[0] = 0.0;
  for (int i = 0; i < N; ++i)
    out[0] += x[i] * y[i];
#pragma endscop
}
