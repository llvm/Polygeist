//@ transform: flatten_1d
//@ extents: A=nr*nq*np, C4=np*np, sum=np
void doitgen__flatten_1d(int nr, int nq, int np, double *A, double *C4, double *sum) {
  for (int r = 0; r < nr; r++)
    for (int q = 0; q < nq; q++) {
      for (int p = 0; p < np; p++) {
        sum[p] = 0.0;
        for (int s = 0; s < np; s++)
          sum[p] += A[(r * nq + q) * np + s] * C4[s * np + p];
      }
      for (int p = 0; p < np; p++)
        A[(r * nq + q) * np + p] = sum[p];
    }
}
