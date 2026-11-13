//@ transform: flatten_1d
//@ function: mm3__flatten_1d
//@ extents: E=ni*nj, A=ni*nk, B=nk*nj, F=nj*nl, C=nj*nm, D=nm*nl, G=ni*nl
void mm3__flatten_1d(int ni, int nj, int nk, int nl, int nm, double *E, double *A, double *B,
                     double *F, double *C, double *D, double *G) {
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++) {
      E[i * nj + j] = 0.0;
      for (int k = 0; k < nk; ++k) E[i * nj + j] += A[i * nk + k] * B[k * nj + j];
    }
  for (int i = 0; i < nj; i++)
    for (int j = 0; j < nl; j++) {
      F[i * nl + j] = 0.0;
      for (int k = 0; k < nm; ++k) F[i * nl + j] += C[i * nm + k] * D[k * nl + j];
    }
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nl; j++) {
      G[i * nl + j] = 0.0;
      for (int k = 0; k < nj; ++k) G[i * nl + j] += E[i * nj + k] * F[k * nl + j];
    }
}
