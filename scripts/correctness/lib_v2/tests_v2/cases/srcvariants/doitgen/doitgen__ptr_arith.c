//@ transform: ptr_arith
void doitgen__ptr_arith(int nr, int nq, int np, double A[nr][nq][np], double C4[np][np], double sum[np]) {
  for (int r = 0; r < nr; r++)
    for (int q = 0; q < nq; q++) {
      double *a = A[r][q];
      for (int p = 0; p < np; p++) {
        sum[p] = 0.0;
        for (int s = 0; s < np; s++)
          sum[p] += a[s] * C4[s][p];
      }
      for (int p = 0; p < np; p++)
        a[p] = sum[p];
    }
}
