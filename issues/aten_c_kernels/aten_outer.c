/* aten::outer numerical body.
 * Upstream family: aten/src/ATen/native/LinearAlgebra.cpp.
 */
#define M 32
#define N 24

void aten_outer(double x[M], double y[N], double out[M][N]) {
#pragma scop
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      out[i][j] = 0.0;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      out[i][j] += x[i] * y[j];
#pragma endscop
}
