#include "3mm.h"


void kernel_3mm(    
		double E[ 40 + 0][50 + 0],
		double A[ 40 + 0][60 + 0],
		double B[ 60 + 0][50 + 0],
		double F[ 50 + 0][70 + 0],
		double C[ 50 + 0][80 + 0],
		double D[ 80 + 0][70 + 0],
		double G[ 40 + 0][70 + 0])
{
  #pragma HLS top name=kernel_3mm

    const int ni = 40;
    const int nj = 50;
    const int nk = 60;
    const int nl = 70;
    const int nm = 80;

  int i, j, k;


  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      {
	E[i][j] = 0.0;
	for (k = 0; k < nk; ++k)
	  E[i][j] += A[i][k] * B[k][j];
      }

  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      {
	F[i][j] = 0.0;
	for (k = 0; k < nm; ++k)
	  F[i][j] += C[i][k] * D[k][j];
      }

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      {
	G[i][j] = 0.0;
	for (k = 0; k < nj; ++k)
	  G[i][j] += E[i][k] * F[k][j];
      }

}