#include "jacobi-1d.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_jacobi_1d(
			    
			    t_ap_fixed A[ 120 + 0],
			    t_ap_fixed B[ 120 + 0])
{
  #pragma HLS top name=kernel_jacobi_1d

    const int n = 120;
    const int tsteps = 40;

  int t, i;

  for (t = 0; t < tsteps; t++)
    {
      for (i = 1; i < n - 1; i++)
	B[i] = (t_ap_fixed(0.33333)) * (A[i-1] + A[i] + A[i + 1]);
      for (i = 1; i < n - 1; i++)
	A[i] = (t_ap_fixed(0.33333)) * (B[i-1] + B[i] + B[i + 1]);
    }

}