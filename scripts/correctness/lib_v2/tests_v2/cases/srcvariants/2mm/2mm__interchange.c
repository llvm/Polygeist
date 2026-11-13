//@ transform: interchange
//@ function: mm2__interchange
// i-k-j order for both products.
void mm2__interchange(int ni, int nj, int nk, int nl, double alpha, double beta, double tmp[ni][nj], double A[ni][nk], double B[nk][nj], double C[nj][nl], double D[ni][nl]) {
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++)
      tmp[i][j] = 0.0;
    for (int k = 0; k < nk; ++k)
      for (int j = 0; j < nj; j++)
        tmp[i][j] += alpha * A[i][k] * B[k][j];
  }
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nl; j++)
      D[i][j] *= beta;
    for (int k = 0; k < nj; ++k)
      for (int j = 0; j < nl; j++)
        D[i][j] += tmp[i][k] * C[k][j];
  }
}
