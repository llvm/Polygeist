#include "lu.h"


void kernel_lu(
	       double A[ 400 + 0][400 + 0])
{
  #pragma HLS top name=kernel_lu

    const int n = 400;

  int i, j, k;

  for (i = 0; i < n; i++) {
    for (j = 0; j <i; j++) {
       for (k = 0; k < j; k++) {
          A[i][j] -= A[i][k] * A[k][j];
       }
        A[i][j] /= A[j][j];
    }
   for (j = i; j < n; j++) {
       for (k = 0; k < i; k++) {
          A[i][j] -= A[i][k] * A[k][j];
       }
    }
  }
}