#include "2mm.h"


void kernel_2mm(   
		double alpha,
		double beta,
		double tmp[ 40 + 0][50 + 0],
		double A[ 40 + 0][70 + 0],
		double B[ 70 + 0][50 + 0],
		double C[ 50 + 0][80 + 0],
		double D[ 40 + 0][80 + 0])
{
  #pragma HLS top name=kernel_2mm

    const int ni = 40;
    const int nj = 50;
    const int nk = 70;
    const int nl = 80;

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