//@ transform: unroll2
#include <math.h>
void cholesky__unroll2(int n, double A[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      int k = 0;
      for (; k + 1 < j; k += 2) {
        A[i][j] -= A[i][k] * A[j][k];
        A[i][j] -= A[i][k + 1] * A[j][k + 1];
      }
      for (; k < j; k++)
        A[i][j] -= A[i][k] * A[j][k];
      A[i][j] /= A[j][j];
    }
    for (int k = 0; k < i; k++)
      A[i][i] -= A[i][k] * A[i][k];
    A[i][i] = sqrt(A[i][i]);
  }
}
