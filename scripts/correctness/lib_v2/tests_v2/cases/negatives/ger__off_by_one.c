//@ routine: ger
//@ kind: off_by_one
//@ expect: nomatch_or_equivalent
// y[j+1] instead of y[j] over j < n-1 (a DGER on shifted operands).
void ger__off_by_one(int m, int n, double alpha, double A[m][n], double x[m], double y[n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n - 1; j++)
      A[i][j] += alpha * x[i] * y[j + 1];
}
