//@ transform: split_reduction
//@ function: mm2__split_reduction
void mm2__split_reduction(int ni, int nj, int nk, int nl, double alpha, double beta, double tmp[ni][nj], double A[ni][nk], double B[nk][nj], double C[nj][nl], double D[ni][nl]) {
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++) {
      double s0 = 0.0, s1 = 0.0;
      for (int k = 0; k < nk / 2; ++k)
        s0 += alpha * A[i][k] * B[k][j];
      for (int k = nk / 2; k < nk; ++k)
        s1 += alpha * A[i][k] * B[k][j];
      tmp[i][j] = s0 + s1;
    }
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nl; j++) {
      D[i][j] *= beta;
      for (int k = 0; k < nj; ++k)
        D[i][j] += tmp[i][k] * C[k][j];
    }
}
