//@ routine: ger
//@ kind: wrong_index
//@ expect: nomatch
//@ square: true
// x[j] instead of x[i].
void ger__wrong_index(int n, double alpha, double A[n][n], double x[n], double y[n]) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      A[i][j] += alpha * x[j] * y[j];
}
