//@ kind: explicit
//@ expect: reject_or_correct
//@ init: spd
// trsv with b aliasing x (in-place forward substitution).
void trsv_b_is_x(int n, double L[n][n], double x[n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
}
