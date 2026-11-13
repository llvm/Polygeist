//@ transform: compound_assign
#include <math.h>
void cholesky__compound_assign(int n, double A[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < j; k++)
        A[i][j] = A[i][j] - A[i][k] * A[j][k];
      A[i][j] = A[i][j] / A[j][j];
    }
    for (int k = 0; k < i; k++)
      A[i][i] = A[i][i] - A[i][k] * A[i][k];
    A[i][i] = sqrt(A[i][i]);
  }
}
