//@ transform: ptr_arith
//@ function: mm3__ptr_arith
void mm3__ptr_arith(int ni, int nj, int nk, int nl, int nm, double E[ni][nj], double A[ni][nk], double B[nk][nj], double F[nj][nl], double C[nj][nm], double D[nm][nl], double G[ni][nl]) {
  for (int i = 0; i < ni; i++) {
    double *e = E[i], *a = A[i];
    for (int j = 0; j < nj; j++) {
      e[j] = 0.0;
      for (int k = 0; k < nk; ++k) e[j] += a[k] * B[k][j];
    }
  }
  for (int i = 0; i < nj; i++) {
    double *f = F[i], *c = C[i];
    for (int j = 0; j < nl; j++) {
      f[j] = 0.0;
      for (int k = 0; k < nm; ++k) f[j] += c[k] * D[k][j];
    }
  }
  for (int i = 0; i < ni; i++) {
    double *g = G[i], *e = E[i];
    for (int j = 0; j < nl; j++) {
      g[j] = 0.0;
      for (int k = 0; k < nj; ++k) g[j] += e[k] * F[k][j];
    }
  }
}
