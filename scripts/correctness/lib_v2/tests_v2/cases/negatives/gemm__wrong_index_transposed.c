//@ routine: gemm
//@ kind: wrong_index
//@ expect: nomatch_or_equivalent
// B[j][k] instead of B[k][j]: this is C := alpha*A*B^T + beta*C, a *different*
// but genuine GEMM (dgemm with a transposed operand).  A match is fine only if
// numerically correct.
void gemm__wrong_index_transposed(int ni, int nj, int nk, double alpha, double beta,
                                  double C[ni][nj], double A[ni][nk], double B[nj][nk]) {
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++)
      C[i][j] *= beta;
    for (int k = 0; k < nk; k++)
      for (int j = 0; j < nj; j++)
        C[i][j] += alpha * A[i][k] * B[j][k];
  }
}
