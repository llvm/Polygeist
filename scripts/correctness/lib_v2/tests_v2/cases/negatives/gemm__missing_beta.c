//@ routine: gemm
//@ kind: missing_beta
//@ expect: nomatch_or_equivalent
// No beta scaling although beta is a parameter: equals dgemm with beta == 1,
// so binding the spec's beta to %beta would be a false positive.
void gemm__missing_beta(int ni, int nj, int nk, double alpha, double beta,
                        double C[ni][nj], double A[ni][nk], double B[nk][nj]) {
  for (int i = 0; i < ni; i++)
    for (int k = 0; k < nk; k++)
      for (int j = 0; j < nj; j++)
        C[i][j] += alpha * A[i][k] * B[k][j];
}
