//@ routine: ger
//@ kind: wrong_triangle
//@ expect: nomatch
//@ square: true
// Only the lower triangle is updated: not a (general) rank-1 update.
void ger__wrong_triangle(int n, double alpha, double A[n][n], double x[n], double y[n]) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j <= i; j++)
      A[i][j] += alpha * x[i] * y[j];
}
