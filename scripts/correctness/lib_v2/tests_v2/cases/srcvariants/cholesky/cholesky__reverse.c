//@ transform: reverse
// Inner reductions run backwards.
#include <math.h>
void cholesky__reverse(int n, double A[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      for (int k = j - 1; k >= 0; k--)
        A[i][j] -= A[i][k] * A[j][k];
      A[i][j] /= A[j][j];
    }
    for (int k = i - 1; k >= 0; k--)
      A[i][i] -= A[i][k] * A[i][k];
    A[i][i] = sqrt(A[i][i]);
  }
}
