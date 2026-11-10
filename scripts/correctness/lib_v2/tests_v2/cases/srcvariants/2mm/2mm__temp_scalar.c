//@ transform: temp_scalar
//@ function: mm2__temp_scalar
void mm2__temp_scalar(int ni, int nj, int nk, int nl, double alpha, double beta, double tmp[ni][nj], double A[ni][nk], double B[nk][nj], double C[nj][nl], double D[ni][nl]) {
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++) {
      double s = 0.0;
      for (int k = 0; k < nk; ++k)
        s += alpha * A[i][k] * B[k][j];
      tmp[i][j] = s;
    }
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nl; j++) {
      double s = beta * D[i][j];
      for (int k = 0; k < nj; ++k)
        s += tmp[i][k] * C[k][j];
      D[i][j] = s;
    }
}
