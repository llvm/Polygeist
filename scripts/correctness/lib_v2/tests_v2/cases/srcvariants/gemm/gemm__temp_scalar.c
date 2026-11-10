//@ transform: temp_scalar
void gemm__temp_scalar(int ni, int nj, int nk, double alpha, double beta, double C[ni][nj], double A[ni][nk], double B[nk][nj]) {
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++) {
      double acc = beta * C[i][j];
      for (int k = 0; k < nk; k++)
        acc += alpha * A[i][k] * B[k][j];
      C[i][j] = acc;
    }
}
