#include "gemm.h"


void kernel_gemm(  
		 double alpha,
		 double beta,
		 double C[ 20 + 0][25 + 0],
		 double A[ 20 + 0][30 + 0],
		 double B[ 30 + 0][25 + 0])
{
  #pragma HLS top name=kernel_gemm

    const int ni = 20;
    const int nj = 25;
    const int nk = 30;

  int i, j, k;
  lp1: for (i = 0; i < ni; i++) {
    lp2: for (j = 0; j < nj; j++)
	C[i][j] *= beta;
    lp3: for (k = 0; k < nk; k++) {
       lp4: for (j = 0; j < nj; j++)
	  C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }

}