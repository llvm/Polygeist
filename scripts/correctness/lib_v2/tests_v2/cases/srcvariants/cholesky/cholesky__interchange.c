//@ transform: interchange
// Column-by-column (Crout / left-looking DPOTF2 order).
#include <math.h>
void cholesky__interchange(int n, double A[n][n]) {
  for (int j = 0; j < n; j++) {
    for (int k = 0; k < j; k++)
      A[j][j] -= A[j][k] * A[j][k];
    A[j][j] = sqrt(A[j][j]);
    for (int i = j + 1; i < n; i++) {
      for (int k = 0; k < j; k++)
        A[i][j] -= A[i][k] * A[j][k];
      A[i][j] /= A[j][j];
    }
  }
}
