/* Positive (leading dimension): L is stored with a leading dimension ld > n;
 * the launch must pass the memory stride, not the used extent. */
void trsv_lda(int n, int ld, double L[n][ld], double x[n], double b[n]) {
  for (int i = 0; i < n; i++) {
    x[i] = b[i];
    for (int j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
}
