//@ transform: interchange
// s outside p: sum[:] accumulated as an axpy sweep.
void doitgen__interchange(int nr, int nq, int np, double A[nr][nq][np], double C4[np][np], double sum[np]) {
  for (int r = 0; r < nr; r++)
    for (int q = 0; q < nq; q++) {
      for (int p = 0; p < np; p++)
        sum[p] = 0.0;
      for (int s = 0; s < np; s++)
        for (int p = 0; p < np; p++)
          sum[p] += A[r][q][s] * C4[s][p];
      for (int p = 0; p < np; p++)
        A[r][q][p] = sum[p];
    }
}
