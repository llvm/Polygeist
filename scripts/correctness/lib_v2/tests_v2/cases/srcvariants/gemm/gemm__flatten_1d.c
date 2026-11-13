//@ transform: flatten_1d
//@ extents: C=ni*nj, A=ni*nk, B=nk*nj
void gemm__flatten_1d(int ni, int nj, int nk, double alpha, double beta,
                      double *C, double *A, double *B) {
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++)
      C[i * nj + j] *= beta;
    for (int k = 0; k < nk; k++)
      for (int j = 0; j < nj; j++)
        C[i * nj + j] += alpha * A[i * nk + k] * B[k * nj + j];
  }
}
