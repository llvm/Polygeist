/* aten::mv accumulation form: y += A@x (beta=1). */
#ifndef M
#define M 64
#endif
#ifndef K
#define K 64
#endif

void aten_mv(double A[M][K], double x[K], double y[M]) {
#pragma scop
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      y[i] += A[i][k] * x[k];
#pragma endscop
}
