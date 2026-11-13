//@ transform: flatten_1d
//@ extents: A=n*n
#include <math.h>
void cholesky__flatten_1d(int n, double *A) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < j; k++)
        A[i * n + j] -= A[i * n + k] * A[j * n + k];
      A[i * n + j] /= A[j * n + j];
    }
    for (int k = 0; k < i; k++)
      A[i * n + i] -= A[i * n + k] * A[i * n + k];
    A[i * n + i] = sqrt(A[i * n + i]);
  }
}
