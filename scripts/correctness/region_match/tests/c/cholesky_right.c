/* Positive: right-looking (outer-product) Cholesky on row-major storage: the
 * trailing lower triangle is updated after each column is finished. */
#include <math.h>
void cholesky_right(int n, double A[n][n]) {
  for (int k = 0; k < n; k++) {
    A[k][k] = sqrt(A[k][k]);
    for (int i = k + 1; i < n; i++)
      A[i][k] = A[i][k] / A[k][k];
    for (int j = k + 1; j < n; j++)
      for (int i = j; i < n; i++)
        A[i][j] -= A[i][k] * A[j][k];
  }
}
