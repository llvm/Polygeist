//@ transform: temp_scalar
void doitgen__temp_scalar(int nr, int nq, int np, double A[nr][nq][np], double C4[np][np], double sum[np]) {
  for (int r = 0; r < nr; r++)
    for (int q = 0; q < nq; q++) {
      for (int p = 0; p < np; p++) {
        double t = 0.0;
        for (int s = 0; s < np; s++)
          t += A[r][q][s] * C4[s][p];
        sum[p] = t;
      }
      for (int p = 0; p < np; p++)
        A[r][q][p] = sum[p];
    }
}
