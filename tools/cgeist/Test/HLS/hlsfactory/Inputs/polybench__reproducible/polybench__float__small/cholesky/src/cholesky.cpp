#include "cholesky.h"


void kernel_cholesky(
		     double A[ 120 + 0][120 + 0])
{
  #pragma HLS top name=kernel_cholesky

    const int n = 120;

  int i, j, k;


  for (i = 0; i < n; i++) {

     for (j = 0; j < i; j++) {
        for (k = 0; k < j; k++) {
           A[i][j] -= A[i][k] * A[j][k];
        }
        A[i][j] /= A[j][j];
     }

     for (k = 0; k < i; k++) {
        A[i][i] -= A[i][k] * A[i][k];
     }
     A[i][i] = sqrt(A[i][i]);
  }

}