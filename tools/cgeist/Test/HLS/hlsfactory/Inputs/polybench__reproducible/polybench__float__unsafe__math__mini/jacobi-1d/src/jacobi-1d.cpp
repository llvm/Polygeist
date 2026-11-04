#include "jacobi-1d.h"


void kernel_jacobi_1d(
			    
			    double A[ 30 + 0],
			    double B[ 30 + 0])
{
  #pragma HLS top name=kernel_jacobi_1d

    const int n = 30;
    const int tsteps = 20;

  int t, i;

  lp1: for (t = 0; t < tsteps; t++)
    {
      lp2: for (i = 1; i < n - 1; i++)
	B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
      lp3: for (i = 1; i < n - 1; i++)
	A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
    }

}