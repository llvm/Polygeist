#include "trmm.h"


void kernel_trmm( 
		 double alpha,
		 double A[ 20 + 0][20 + 0],
		 double B[ 20 + 0][30 + 0])
{
  #pragma HLS top name=kernel_trmm

    const int m = 20;
    const int n = 30;

  int i, j, k;
  lp1: for (i = 0; i < m; i++)
     lp2: for (j = 0; j < n; j++) {
        lp3: for (k = i+1; k < m; k++)
           B[i][j] += A[k][i] * B[k][j];
        B[i][j] = alpha * B[i][j];
     }

}