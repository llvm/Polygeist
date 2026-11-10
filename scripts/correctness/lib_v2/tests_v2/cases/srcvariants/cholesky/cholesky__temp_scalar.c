//@ transform: temp_scalar
#include <math.h>
void cholesky__temp_scalar(int n, double A[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      double s = A[i][j];
      for (int k = 0; k < j; k++)
        s -= A[i][k] * A[j][k];
      A[i][j] = s / A[j][j];
    }
    double d = A[i][i];
    for (int k = 0; k < i; k++)
      d -= A[i][k] * A[i][k];
    A[i][i] = sqrt(d);
  }
}
