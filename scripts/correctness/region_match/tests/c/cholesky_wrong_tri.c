/* Negative: PolyBench cholesky but the dot product reads A[k][j] (upper,
 * never computed) instead of A[j][k]. */
#include <math.h>
void cholesky_wrong_tri(int n, double A[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < j; k++)
        A[i][j] -= A[i][k] * A[k][j];
      A[i][j] /= A[j][j];
    }
    for (int k = 0; k < i; k++)
      A[i][i] -= A[i][k] * A[i][k];
    A[i][i] = sqrt(A[i][i]);
  }
}
