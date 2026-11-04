#include "gemver.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_gemver(
		   t_ap_fixed alpha,
		   t_ap_fixed beta,
		   t_ap_fixed A[ 40 + 0][40 + 0],
		   t_ap_fixed u1[ 40 + 0],
		   t_ap_fixed v1[ 40 + 0],
		   t_ap_fixed u2[ 40 + 0],
		   t_ap_fixed v2[ 40 + 0],
		   t_ap_fixed w[ 40 + 0],
		   t_ap_fixed x[ 40 + 0],
		   t_ap_fixed y[ 40 + 0],
		   t_ap_fixed z[ 40 + 0])
{
  #pragma HLS top name=kernel_gemver

    const int n = 40;

  int i, j;


  lp1: for (i = 0; i < n; i++)
    lp2: for (j = 0; j < n; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  lp3: for (i = 0; i < n; i++)
    lp4: for (j = 0; j < n; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  lp5: for (i = 0; i < n; i++)
    x[i] = x[i] + z[i];

  lp6: for (i = 0; i < n; i++)
    lp7: for (j = 0; j < n; j++)
      w[i] = w[i] +  alpha * A[i][j] * x[j];

}