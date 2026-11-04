#include "atax.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_atax( 
		 t_ap_fixed A[ 116 + 0][124 + 0],
		 t_ap_fixed x[ 124 + 0],
		 t_ap_fixed y[ 124 + 0],
		 t_ap_fixed tmp[ 116 + 0])
{
  #pragma HLS top name=kernel_atax

    const int m = 116;
    const int n = 124;

  int i, j;

  for (i = 0; i < n; i++)
    y[i] = 0;
  for (i = 0; i < m; i++)
    {
      tmp[i] = (t_ap_fixed(0.0));
      for (j = 0; j < n; j++)
	tmp[i] = tmp[i] + A[i][j] * x[j];
      for (j = 0; j < n; j++)
	y[j] = y[j] + A[i][j] * tmp[i];
    }

}