//@ kind: explicit
//@ expect: reject_or_correct
//@ init: spd
// trsv with b = x + 1 (overlapping, shifted): x[i] = (x[i+1] - sum) / L[i][i].
void trsv_b_overlap_x(int n, double L[n][n], double x[n + 1]) {
  for (int i = 0; i < n; i++) {
    x[i] = x[i + 1];
    for (int j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
}
