#include "bicg.h"


void kernel_bicg( 
		 double A[ 410 + 0][390 + 0],
		 double s[ 390 + 0],
		 double q[ 410 + 0],
		 double p[ 390 + 0],
		 double r[ 410 + 0])
{
  #pragma HLS top name=kernel_bicg

    const int n = 410;
    const int m = 390;

  int i, j;

  for (i = 0; i < m; i++)
    s[i] = 0;
  for (i = 0; i < n; i++)
    {
      q[i] = 0.0;
      for (j = 0; j < m; j++)
	{
	  s[j] = s[j] + r[i] * A[i][j];
	  q[i] = q[i] + A[i][j] * p[j];
	}
    }

}