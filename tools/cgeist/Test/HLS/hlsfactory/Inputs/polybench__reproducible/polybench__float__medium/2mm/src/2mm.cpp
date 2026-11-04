#include "2mm.h"


void kernel_2mm(   
		double alpha,
		double beta,
		double tmp[ 180 + 0][190 + 0],
		double A[ 180 + 0][210 + 0],
		double B[ 210 + 0][190 + 0],
		double C[ 190 + 0][220 + 0],
		double D[ 180 + 0][220 + 0])
{
  #pragma HLS top name=kernel_2mm

    const int ni = 180;
    const int nj = 190;
    const int nk = 210;
    const int nl = 220;

  int i, j, k;


  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      {
	tmp[i][j] = 0.0;
	for (k = 0; k < nk; ++k)
	  tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      {
	D[i][j] *= beta;
	for (k = 0; k < nj; ++k)
	  D[i][j] += tmp[i][k] * C[k][j];
      }

}