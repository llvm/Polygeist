//@ transform: unroll2
void doitgen__unroll2(int nr, int nq, int np, double A[nr][nq][np], double C4[np][np], double sum[np]) {
  for (int r = 0; r < nr; r++)
    for (int q = 0; q < nq; q++) {
      for (int p = 0; p < np; p++) {
        sum[p] = 0.0;
        int s = 0;
        for (; s + 1 < np; s += 2) {
          sum[p] += A[r][q][s] * C4[s][p];
          sum[p] += A[r][q][s + 1] * C4[s + 1][p];
        }
        for (; s < np; s++)
          sum[p] += A[r][q][s] * C4[s][p];
      }
      for (int p = 0; p < np; p++)
        A[r][q][p] = sum[p];
    }
}
