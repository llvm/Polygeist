#include "mvt.h"


void kernel_mvt(
		double x1[ 120 + 0],
		double x2[ 120 + 0],
		double y_1[ 120 + 0],
		double y_2[ 120 + 0],
		double A[ 120 + 0][120 + 0])
{
  #pragma HLS top name=kernel_mvt

    const int n = 120;

  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];

}