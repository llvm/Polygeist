//@ routine: gemm
//@ kind: off_by_one
//@ expect: nomatch
// Accumulation stops at nj-1 while the beta scaling covers nj columns.
void gemm__off_by_one(int ni, int nj, int nk, double alpha, double beta,
                      double C[ni][nj], double A[ni][nk], double B[nk][nj]) {
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++)
      C[i][j] *= beta;
    for (int k = 0; k < nk; k++)
      for (int j = 0; j < nj - 1; j++)
        C[i][j] += alpha * A[i][k] * B[k][j];
  }
}
