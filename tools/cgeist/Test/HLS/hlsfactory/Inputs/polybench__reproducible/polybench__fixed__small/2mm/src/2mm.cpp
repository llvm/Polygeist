#include "2mm.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_2mm(   
		t_ap_fixed alpha,
		t_ap_fixed beta,
		t_ap_fixed tmp[ 40 + 0][50 + 0],
		t_ap_fixed A[ 40 + 0][70 + 0],
		t_ap_fixed B[ 70 + 0][50 + 0],
		t_ap_fixed C[ 50 + 0][80 + 0],
		t_ap_fixed D[ 40 + 0][80 + 0])
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
	tmp[i][j] = (t_ap_fixed(0.0));
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