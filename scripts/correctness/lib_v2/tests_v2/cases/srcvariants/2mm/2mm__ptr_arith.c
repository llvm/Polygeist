//@ transform: ptr_arith
//@ function: mm2__ptr_arith
void mm2__ptr_arith(int ni, int nj, int nk, int nl, double alpha, double beta, double tmp[ni][nj], double A[ni][nk], double B[nk][nj], double C[nj][nl], double D[ni][nl]) {
  for (int i = 0; i < ni; i++) {
    double *t = tmp[i];
    double *a = A[i];
    for (int j = 0; j < nj; j++) {
      *(t + j) = 0.0;
      for (int k = 0; k < nk; ++k)
        *(t + j) += alpha * *(a + k) * B[k][j];
    }
  }
  for (int i = 0; i < ni; i++) {
    double *d = D[i];
    double *t = tmp[i];
    for (int j = 0; j < nl; j++) {
      d[j] *= beta;
      for (int k = 0; k < nj; ++k)
        d[j] += t[k] * C[k][j];
    }
  }
}
