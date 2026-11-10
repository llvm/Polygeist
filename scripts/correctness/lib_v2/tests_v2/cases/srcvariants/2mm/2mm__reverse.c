//@ transform: reverse
//@ function: mm2__reverse
void mm2__reverse(int ni, int nj, int nk, int nl, double alpha, double beta, double tmp[ni][nj], double A[ni][nk], double B[nk][nj], double C[nj][nl], double D[ni][nl]) {
  for (int i = ni - 1; i >= 0; i--)
    for (int j = nj - 1; j >= 0; j--) {
      tmp[i][j] = 0.0;
      for (int k = 0; k < nk; ++k)
        tmp[i][j] += alpha * A[i][k] * B[k][j];
    }
  for (int i = ni - 1; i >= 0; i--)
    for (int j = nl - 1; j >= 0; j--) {
      D[i][j] *= beta;
      for (int k = 0; k < nj; ++k)
        D[i][j] += tmp[i][k] * C[k][j];
    }
}
