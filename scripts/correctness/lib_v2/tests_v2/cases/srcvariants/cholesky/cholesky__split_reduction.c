//@ transform: split_reduction
#include <math.h>
void cholesky__split_reduction(int n, double A[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      double s0 = 0.0, s1 = 0.0;
      for (int k = 0; k < j / 2; k++)
        s0 += A[i][k] * A[j][k];
      for (int k = j / 2; k < j; k++)
        s1 += A[i][k] * A[j][k];
      A[i][j] = (A[i][j] - (s0 + s1)) / A[j][j];
    }
    double s0 = 0.0, s1 = 0.0;
    for (int k = 0; k < i / 2; k++)
      s0 += A[i][k] * A[i][k];
    for (int k = i / 2; k < i; k++)
      s1 += A[i][k] * A[i][k];
    A[i][i] = sqrt(A[i][i] - (s0 + s1));
  }
}
