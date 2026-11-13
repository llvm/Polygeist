//@ transform: reverse
//@ function: mm3__reverse
void mm3__reverse(int ni, int nj, int nk, int nl, int nm, double E[ni][nj], double A[ni][nk], double B[nk][nj], double F[nj][nl], double C[nj][nm], double D[nm][nl], double G[ni][nl]) {
  for (int i = ni - 1; i >= 0; i--)
    for (int j = nj - 1; j >= 0; j--) {
      E[i][j] = 0.0;
      for (int k = 0; k < nk; ++k) E[i][j] += A[i][k] * B[k][j];
    }
  for (int i = nj - 1; i >= 0; i--)
    for (int j = nl - 1; j >= 0; j--) {
      F[i][j] = 0.0;
      for (int k = 0; k < nm; ++k) F[i][j] += C[i][k] * D[k][j];
    }
  for (int i = ni - 1; i >= 0; i--)
    for (int j = nl - 1; j >= 0; j--) {
      G[i][j] = 0.0;
      for (int k = 0; k < nj; ++k) G[i][j] += E[i][k] * F[k][j];
    }
}
