#include "trmm.h"


void kernel_trmm( 
		 double alpha,
		 double A[ 200 + 0][200 + 0],
		 double B[ 200 + 0][240 + 0])
{
  #pragma HLS top name=kernel_trmm

    const int m = 200;
    const int n = 240;

  int i, j, k;
  for (i = 0; i < m; i++)
     for (j = 0; j < n; j++) {
        for (k = i+1; k < m; k++)
           B[i][j] += A[k][i] * B[k][j];
        B[i][j] = alpha * B[i][j];
     }

}