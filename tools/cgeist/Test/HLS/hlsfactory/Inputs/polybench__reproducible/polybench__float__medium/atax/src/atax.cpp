#include "atax.h"


void kernel_atax( 
		 double A[ 390 + 0][410 + 0],
		 double x[ 410 + 0],
		 double y[ 410 + 0],
		 double tmp[ 390 + 0])
{
  #pragma HLS top name=kernel_atax

    const int m = 390;
    const int n = 410;

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