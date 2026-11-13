//@ transform: reverse
void doitgen__reverse(int nr, int nq, int np, double A[nr][nq][np], double C4[np][np], double sum[np]) {
  for (int r = nr - 1; r >= 0; r--)
    for (int q = nq - 1; q >= 0; q--) {
      for (int p = np - 1; p >= 0; p--) {
        sum[p] = 0.0;
        for (int s = 0; s < np; s++)
          sum[p] += A[r][q][s] * C4[s][p];
      }
      for (int p = np - 1; p >= 0; p--)
        A[r][q][p] = sum[p];
    }
}
