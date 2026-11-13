/* Positive: column-by-column (left-looking, Crout-like) Cholesky on row-major
 * storage: column j is finished before column j+1 starts. */
#include <math.h>
void cholesky_crout(int n, double A[n][n]) {
  for (int j = 0; j < n; j++) {
    for (int k = 0; k < j; k++)
      A[j][j] -= A[j][k] * A[j][k];
    A[j][j] = sqrt(A[j][j]);
    for (int i = j + 1; i < n; i++) {
      for (int k = 0; k < j; k++)
        A[i][j] -= A[i][k] * A[j][k];
      A[i][j] = A[i][j] / A[j][j];
    }
  }
}
