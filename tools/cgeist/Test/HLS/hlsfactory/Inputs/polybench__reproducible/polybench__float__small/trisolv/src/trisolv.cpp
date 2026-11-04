#include "trisolv.h"


void kernel_trisolv(
		    double L[ 120 + 0][120 + 0],
		    double x[ 120 + 0],
		    double b[ 120 + 0])
{
  #pragma HLS top name=kernel_trisolv

    const int n = 120;

  int i, j;

  for (i = 0; i < n; i++)
    {
      x[i] = b[i];
      for (j = 0; j <i; j++)
        x[i] -= L[i][j] * x[j];
      x[i] = x[i] / L[i][i];
    }

}