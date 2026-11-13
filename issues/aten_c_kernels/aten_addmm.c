/* aten::addmm: C = beta*C + alpha*(A@B). */
#ifndef M
#define M 16
#endif
#ifndef N
#define N 16
#endif
#ifndef K
#define K 16
#endif

void aten_addmm(double A[M][K], double B[K][N], double C[M][N],
                double beta, double alpha) {
#pragma scop
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      C[i][j] *= beta;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < K; ++k)
        C[i][j] += alpha * A[i][k] * B[k][j];
#pragma endscop
}
