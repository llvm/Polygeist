#include "atax.h"


void kernel_atax( 
		 double A[ 116 + 0][124 + 0],
		 double x[ 124 + 0],
		 double y[ 124 + 0],
		 double tmp[ 116 + 0])
{
  #pragma HLS top name=kernel_atax

    const int m = 116;
    const int n = 124;

  int i, j;

  for (i = 0; i < n; i++)
    y[i] = 0;
  for (i = 0; i < m; i++)
    {
      tmp[i] = 0.0;
      for (j = 0; j < n; j++)
	tmp[i] = tmp[i] + A[i][j] * x[j];
      for (j = 0; j < n; j++)
	y[j] = y[j] + A[i][j] * tmp[i];
    }

}