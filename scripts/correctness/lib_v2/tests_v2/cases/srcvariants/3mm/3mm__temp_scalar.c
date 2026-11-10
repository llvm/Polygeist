//@ transform: temp_scalar
//@ function: mm3__temp_scalar
void mm3__temp_scalar(int ni, int nj, int nk, int nl, int nm, double E[ni][nj], double A[ni][nk], double B[nk][nj], double F[nj][nl], double C[nj][nm], double D[nm][nl], double G[ni][nl]) {
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++) {
      double s = 0.0;
      for (int k = 0; k < nk; ++k) s += A[i][k] * B[k][j];
      E[i][j] = s;
    }
  for (int i = 0; i < nj; i++)
    for (int j = 0; j < nl; j++) {
      double s = 0.0;
      for (int k = 0; k < nm; ++k) s += C[i][k] * D[k][j];
      F[i][j] = s;
    }
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nl; j++) {
      double s = 0.0;
      for (int k = 0; k < nj; ++k) s += E[i][k] * F[k][j];
      G[i][j] = s;
    }
}
