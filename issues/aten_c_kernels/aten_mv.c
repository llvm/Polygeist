/* aten::mv accumulation form: y += A@x (beta=1). */
#define M 64
#define K 64

void aten_mv(double A[M][K], double x[K], double y[M]) {
#pragma scop
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      y[i] += A[i][k] * x[k];
#pragma endscop
}
