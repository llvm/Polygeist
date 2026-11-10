/* Positive + policy negative: C += A*B.  The general-beta DGEMM spec assumes
 * beta != 0 and beta != 1, so beta must not be bound to the literal 1.0; the
 * beta1 specialisation (alpha bound to 1.0) is the match. */
void gemm_beta1(int ni, int nj, int nk, double C[ni][nj], double A[ni][nk], double B[nk][nj]) {
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++)
      for (int k = 0; k < nk; k++)
        C[i][j] += A[i][k] * B[k][j];
}
