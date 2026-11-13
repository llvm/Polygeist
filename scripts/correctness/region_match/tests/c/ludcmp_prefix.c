/* Not a matcher input: the code of PolyBench kernel_ludcmp that precedes the
 * region the matcher replaces (LU factorisation + forward substitution,
 * verbatim).  numeric_check runs it on the spec side before the emitted
 * launch, so a mid-function match is checked against the whole kernel. */
void kernel_ludcmp_prefix(int n, double A[n][n], double b[n], double x[n], double y[n]) {
  int i, j, k;
  double w;
  for (i = 0; i < n; i++) {
    for (j = 0; j < i; j++) {
      w = A[i][j];
      for (k = 0; k < j; k++)
        w -= A[i][k] * A[k][j];
      A[i][j] = w / A[j][j];
    }
    for (j = i; j < n; j++) {
      w = A[i][j];
      for (k = 0; k < i; k++)
        w -= A[i][k] * A[k][j];
      A[i][j] = w;
    }
  }
  for (i = 0; i < n; i++) {
    w = b[i];
    for (j = 0; j < i; j++)
      w -= A[i][j] * y[j];
    y[i] = w;
  }
}
