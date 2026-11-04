#include "bicg.h"


void kernel_bicg( 
		 double A[ 42 + 0][38 + 0],
		 double s[ 38 + 0],
		 double q[ 42 + 0],
		 double p[ 38 + 0],
		 double r[ 42 + 0])
{
  #pragma HLS top name=kernel_bicg

    const int n = 42;
    const int m = 38;

  int i, j;

  lp0: for (i = 0; i < m; i++)
    s[i] = 0;
  lp1: for (i = 0; i < n; i++)
    {
      q[i] = 0.0;
      lp2: for (j = 0; j < m; j++)
	{
	  s[j] = s[j] + r[i] * A[i][j];
	  q[i] = q[i] + A[i][j] * p[j];
	}
    }

}