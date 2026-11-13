//@ transform: split_reduction
// The k reduction is split into two partial sums over halves of the range.
void gemm__split_reduction(int ni, int nj, int nk, double alpha, double beta, double C[ni][nj], double A[ni][nk], double B[nk][nj]) {
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++) {
      double s0 = 0.0, s1 = 0.0;
      for (int k = 0; k < nk / 2; k++)
        s0 += A[i][k] * B[k][j];
      for (int k = nk / 2; k < nk; k++)
        s1 += A[i][k] * B[k][j];
      C[i][j] = beta * C[i][j] + alpha * (s0 + s1);
    }
}
