#include "gemm.h"


void kernel_gemm(  
		 double alpha,
		 double beta,
		 double C[ 60 + 0][70 + 0],
		 double A[ 60 + 0][80 + 0],
		 double B[ 80 + 0][70 + 0])
{
  #pragma HLS top name=kernel_gemm

    const int ni = 60;
    const int nj = 70;
    const int nk = 80;

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