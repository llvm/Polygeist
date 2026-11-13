//@ transform: alpha_fold
// alpha folded into a hoisted per-(i,k) scalar (reference-BLAS style).
void gemm__alpha_fold(int ni, int nj, int nk, double alpha, double beta, double C[ni][nj], double A[ni][nk], double B[nk][nj]) {
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++)
      C[i][j] *= beta;
    for (int k = 0; k < nk; k++) {
      double t = alpha * A[i][k];
      for (int j = 0; j < nj; j++)
        C[i][j] += t * B[k][j];
    }
  }
}
