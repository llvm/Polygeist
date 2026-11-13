//@ transform: compound_assign
//@ function: mm2__compound_assign
void mm2__compound_assign(int ni, int nj, int nk, int nl, double alpha, double beta, double tmp[ni][nj], double A[ni][nk], double B[nk][nj], double C[nj][nl], double D[ni][nl]) {
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++) {
      tmp[i][j] = 0.0;
      for (int k = 0; k < nk; ++k)
        tmp[i][j] = tmp[i][j] + alpha * A[i][k] * B[k][j];
    }
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nl; j++) {
      D[i][j] = D[i][j] * beta;
      for (int k = 0; k < nj; ++k)
        D[i][j] = D[i][j] + tmp[i][k] * C[k][j];
    }
}
