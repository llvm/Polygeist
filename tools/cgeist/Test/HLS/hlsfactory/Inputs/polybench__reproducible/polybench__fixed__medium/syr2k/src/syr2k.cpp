#include "syr2k.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_syr2k( 
		  t_ap_fixed alpha,
		  t_ap_fixed beta,
		  t_ap_fixed C[ 240 + 0][240 + 0],
		  t_ap_fixed A[ 240 + 0][200 + 0],
		  t_ap_fixed B[ 240 + 0][200 + 0])
{
  #pragma HLS top name=kernel_syr2k

    const int n = 240;
    const int m = 200;

  int i, j, k;
  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < m; k++)
      for (j = 0; j <= i; j++)
	{
	  C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k];
	}
  }

}