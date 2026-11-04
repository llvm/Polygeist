#include "seidel-2d.h"


void kernel_seidel_2d(
		      
		      double A[ 40 + 0][40 + 0])
{
  #pragma HLS top name=kernel_seidel_2d

    const int n = 40;
    const int tsteps = 20;

  int t, i, j;

  lp1: for (t = 0; t <= tsteps - 1; t++)
    lp2: for (i = 1; i<= n - 2; i++)
      lp3: for (j = 1; j <= n - 2; j++)
	A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
		   + A[i][j-1] + A[i][j] + A[i][j+1]
		   + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/9.0;

}