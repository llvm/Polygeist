#include "gesummv.h"


void kernel_gesummv(
		    double alpha,
		    double beta,
		    double A[ 30 + 0][30 + 0],
		    double B[ 30 + 0][30 + 0],
		    double tmp[ 30 + 0],
		    double x[ 30 + 0],
		    double y[ 30 + 0])
{
  #pragma HLS top name=kernel_gesummv

    const int n = 30;

  int i, j;

  lp1: for (i = 0; i < n; i++)
    {
      tmp[i] = 0.0;
      y[i] = 0.0;
      lp2: for (j = 0; j < n; j++)
	{
	  tmp[i] = A[i][j] * x[j] + tmp[i];
	  y[i] = B[i][j] * x[j] + y[i];
	}
      y[i] = alpha * tmp[i] + beta * y[i];
    }

}