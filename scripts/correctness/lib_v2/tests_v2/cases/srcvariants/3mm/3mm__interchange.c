//@ transform: interchange
//@ function: mm3__interchange
void mm3__interchange(int ni, int nj, int nk, int nl, int nm, double E[ni][nj], double A[ni][nk], double B[nk][nj], double F[nj][nl], double C[nj][nm], double D[nm][nl], double G[ni][nl]) {
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) E[i][j] = 0.0;
    for (int k = 0; k < nk; ++k)
      for (int j = 0; j < nj; j++) E[i][j] += A[i][k] * B[k][j];
  }
  for (int i = 0; i < nj; i++) {
    for (int j = 0; j < nl; j++) F[i][j] = 0.0;
    for (int k = 0; k < nm; ++k)
      for (int j = 0; j < nl; j++) F[i][j] += C[i][k] * D[k][j];
  }
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nl; j++) G[i][j] = 0.0;
    for (int k = 0; k < nj; ++k)
      for (int j = 0; j < nl; j++) G[i][j] += E[i][k] * F[k][j];
  }
}
