//@ transform: unroll2
//@ function: mm3__unroll2
// j unrolled by two (remainder handled) in every product.
void mm3__unroll2(int ni, int nj, int nk, int nl, int nm, double E[ni][nj], double A[ni][nk], double B[nk][nj], double F[nj][nl], double C[nj][nm], double D[nm][nl], double G[ni][nl]) {
  for (int i = 0; i < ni; i++) {
    int j = 0;
    for (; j + 1 < nj; j += 2) {
      E[i][j] = 0.0; E[i][j + 1] = 0.0;
      for (int k = 0; k < nk; ++k) { E[i][j] += A[i][k] * B[k][j]; E[i][j + 1] += A[i][k] * B[k][j + 1]; }
    }
    for (; j < nj; j++) { E[i][j] = 0.0; for (int k = 0; k < nk; ++k) E[i][j] += A[i][k] * B[k][j]; }
  }
  for (int i = 0; i < nj; i++) {
    int j = 0;
    for (; j + 1 < nl; j += 2) {
      F[i][j] = 0.0; F[i][j + 1] = 0.0;
      for (int k = 0; k < nm; ++k) { F[i][j] += C[i][k] * D[k][j]; F[i][j + 1] += C[i][k] * D[k][j + 1]; }
    }
    for (; j < nl; j++) { F[i][j] = 0.0; for (int k = 0; k < nm; ++k) F[i][j] += C[i][k] * D[k][j]; }
  }
  for (int i = 0; i < ni; i++) {
    int j = 0;
    for (; j + 1 < nl; j += 2) {
      G[i][j] = 0.0; G[i][j + 1] = 0.0;
      for (int k = 0; k < nj; ++k) { G[i][j] += E[i][k] * F[k][j]; G[i][j + 1] += E[i][k] * F[k][j + 1]; }
    }
    for (; j < nl; j++) { G[i][j] = 0.0; for (int k = 0; k < nj; ++k) G[i][j] += E[i][k] * F[k][j]; }
  }
}
