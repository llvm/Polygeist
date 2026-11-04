#include "3mm.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_3mm(    
		t_ap_fixed E[ 40 + 0][50 + 0],
		t_ap_fixed A[ 40 + 0][60 + 0],
		t_ap_fixed B[ 60 + 0][50 + 0],
		t_ap_fixed F[ 50 + 0][70 + 0],
		t_ap_fixed C[ 50 + 0][80 + 0],
		t_ap_fixed D[ 80 + 0][70 + 0],
		t_ap_fixed G[ 40 + 0][70 + 0])
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
	E[i][j] = (t_ap_fixed(0.0));
	for (k = 0; k < nk; ++k)
	  E[i][j] += A[i][k] * B[k][j];
      }

  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      {
	F[i][j] = (t_ap_fixed(0.0));
	for (k = 0; k < nm; ++k)
	  F[i][j] += C[i][k] * D[k][j];
      }

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      {
	G[i][j] = (t_ap_fixed(0.0));
	for (k = 0; k < nj; ++k)
	  G[i][j] += E[i][k] * F[k][j];
      }

}