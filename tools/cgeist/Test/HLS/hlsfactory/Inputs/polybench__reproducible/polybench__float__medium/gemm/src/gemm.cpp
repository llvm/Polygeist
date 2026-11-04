#include "gemm.h"


void kernel_gemm(  
		 double alpha,
		 double beta,
		 double C[ 200 + 0][220 + 0],
		 double A[ 200 + 0][240 + 0],
		 double B[ 240 + 0][220 + 0])
{
  #pragma HLS top name=kernel_gemm

    const int ni = 200;
    const int nj = 220;
    const int nk = 240;

  int i, j, k;
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++)
	C[i][j] *= beta;
    for (k = 0; k < nk; k++) {
       for (j = 0; j < nj; j++)
	  C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }

}