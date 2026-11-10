/* Positive: row-major upper Cholesky A = U^T U with U in the upper triangle
 * (U[j][i], i >= j), column-oriented.  Same memory layout as LAPACK UPLO='L'
 * on column-major storage, so it must bind to netlib_dpotf2_l without a
 * transpose. */
#include <math.h>
void cholesky_upper(int n, double A[n][n]) {
  for (int j = 0; j < n; j++) {
    for (int k = 0; k < j; k++)
      A[j][j] -= A[k][j] * A[k][j];
    A[j][j] = sqrt(A[j][j]);
    for (int i = j + 1; i < n; i++) {
      for (int k = 0; k < j; k++)
        A[j][i] -= A[k][j] * A[k][i];
      A[j][i] = A[j][i] / A[j][j];
    }
  }
}
