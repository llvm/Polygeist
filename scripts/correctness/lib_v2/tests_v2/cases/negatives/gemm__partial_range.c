//@ routine: gemm
//@ kind: partial_range
//@ expect: nomatch_or_equivalent
// Row 0 of C is left untouched (a GEMM on the sub-block rows 1..ni-1 only).
void gemm__partial_range(int ni, int nj, int nk, double alpha, double beta,
                         double C[ni][nj], double A[ni][nk], double B[nk][nj]) {
  for (int i = 1; i < ni; i++) {
    for (int j = 0; j < nj; j++)
      C[i][j] *= beta;
    for (int k = 0; k < nk; k++)
      for (int j = 0; j < nj; j++)
        C[i][j] += alpha * A[i][k] * B[k][j];
  }
}
