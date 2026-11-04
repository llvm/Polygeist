#include "jacobi-2d.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_jacobi_2d(
			    
			    t_ap_fixed A[ 30 + 0][30 + 0],
			    t_ap_fixed B[ 30 + 0][30 + 0])
{
  #pragma HLS top name=kernel_jacobi_2d

    const int n = 30;
    const int tsteps = 20;

  int t, i, j;

  lp1: for (t = 0; t < tsteps; t++)
    {
      lp2: for (i = 1; i < n - 1; i++)
	lp3: for (j = 1; j < n - 1; j++)
	  B[i][j] = (t_ap_fixed(0.2)) * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
      lp4: for (i = 1; i < n - 1; i++)
	lp5: for (j = 1; j < n - 1; j++)
	  A[i][j] = (t_ap_fixed(0.2)) * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
    }

}