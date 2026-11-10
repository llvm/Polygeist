//@ transform: split_reduction
void doitgen__split_reduction(int nr, int nq, int np, double A[nr][nq][np], double C4[np][np], double sum[np]) {
  for (int r = 0; r < nr; r++)
    for (int q = 0; q < nq; q++) {
      for (int p = 0; p < np; p++) {
        double s0 = 0.0, s1 = 0.0;
        for (int s = 0; s < np / 2; s++)
          s0 += A[r][q][s] * C4[s][p];
        for (int s = np / 2; s < np; s++)
          s1 += A[r][q][s] * C4[s][p];
        sum[p] = s0 + s1;
      }
      for (int p = 0; p < np; p++)
        A[r][q][p] = sum[p];
    }
}
