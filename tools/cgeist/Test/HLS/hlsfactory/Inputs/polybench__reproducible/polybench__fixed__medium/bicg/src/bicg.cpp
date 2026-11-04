#include "bicg.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_bicg( 
		 t_ap_fixed A[ 410 + 0][390 + 0],
		 t_ap_fixed s[ 390 + 0],
		 t_ap_fixed q[ 410 + 0],
		 t_ap_fixed p[ 390 + 0],
		 t_ap_fixed r[ 410 + 0])
{
  #pragma HLS top name=kernel_bicg

    const int n = 410;
    const int m = 390;

  int i, j;

  for (i = 0; i < m; i++)
    s[i] = 0;
  for (i = 0; i < n; i++)
    {
      q[i] = (t_ap_fixed(0.0));
      for (j = 0; j < m; j++)
	{
	  s[j] = s[j] + r[i] * A[i][j];
	  q[i] = q[i] + A[i][j] * p[j];
	}
    }

}