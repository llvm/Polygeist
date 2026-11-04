#include "symm.h"


void kernel_symm( 
		 double alpha,
		 double beta,
		 double C[ 20 + 0][30 + 0],
		 double A[ 20 + 0][20 + 0],
		 double B[ 20 + 0][30 + 0])
{
  #pragma HLS top name=kernel_symm

    const int m = 20;
    const int n = 30;

  int i, j, k;
  double temp2;
   lp1: for (i = 0; i < m; i++)
      lp2: for (j = 0; j < n; j++ )
      {
        temp2 = 0;
        lp3: for (k = 0; k < i; k++) {
           C[k][j] += alpha*B[i][j] * A[i][k];
           temp2 += B[k][j] * A[i][k];
        }
        C[i][j] = beta * C[i][j] + alpha*B[i][j] * A[i][i] + alpha * temp2;
     }

}