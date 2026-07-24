/* aten::sum over the final dimension.
 * Upstream family: aten/src/ATen/native/ReduceOps.cpp.
 */
#define M 16
#define N 64

void aten_sum(double x[M][N], double out[M]) {
#pragma scop
  for (int i = 0; i < M; ++i)
    out[i] = 0.0;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      out[i] += x[i][j];
#pragma endscop
}
