/* Positive (composition): rank-2 update = DGER ; DGER (alpha bound to 1.0). */
void rank2(int n, double A[n][n], double u1[n], double v1[n], double u2[n], double v2[n]) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
}
