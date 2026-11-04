#include "bicg.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_bicg( 
		 t_ap_fixed A[ 42 + 0][38 + 0],
		 t_ap_fixed s[ 38 + 0],
		 t_ap_fixed q[ 42 + 0],
		 t_ap_fixed p[ 38 + 0],
		 t_ap_fixed r[ 42 + 0])
{
  #pragma HLS top name=kernel_bicg

    const int n = 42;
    const int m = 38;

  int i, j;

  lp0: for (i = 0; i < m; i++)
    s[i] = 0;
  lp1: for (i = 0; i < n; i++)
    {
      q[i] = (t_ap_fixed(0.0));
      lp2: for (j = 0; j < m; j++)
	{
	  s[j] = s[j] + r[i] * A[i][j];
	  q[i] = q[i] + A[i][j] * p[j];
	}
    }

}