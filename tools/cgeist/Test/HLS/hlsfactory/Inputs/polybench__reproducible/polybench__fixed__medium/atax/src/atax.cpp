#include "atax.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_atax( 
		 t_ap_fixed A[ 390 + 0][410 + 0],
		 t_ap_fixed x[ 410 + 0],
		 t_ap_fixed y[ 410 + 0],
		 t_ap_fixed tmp[ 390 + 0])
{
  #pragma HLS top name=kernel_atax

    const int m = 390;
    const int n = 410;

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