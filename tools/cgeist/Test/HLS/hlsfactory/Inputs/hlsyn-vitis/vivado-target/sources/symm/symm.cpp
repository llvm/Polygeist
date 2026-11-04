#include "symm.h"
void kernel_symm(double alpha, double beta, double C[60][80], double A[60][60],
                 double B[60][80]) {
  double C_buf[60][80];
  double A_buf[60][60];
  double B_buf[60][80];
  memcpy(C_buf, C, sizeof(double) * 60 * 80);
  memcpy(A_buf, A, sizeof(double) * 60 * 60);
  memcpy(B_buf, B, sizeof(double) * 60 * 80);

  int i;
  int j;
  int k;
  // BLAS PARAMS
  // SIDE = 'L'
  // UPLO = 'L'
  //  =>  Form  C := alpha*A*B + beta*C
  //  A is MxM
  //  B is MxN
  //  C is MxN
  // note that due to Fortran array layout, the code below more closely
  // resembles upper triangular case in BLAS

  L0: for (i = 0; i < 60; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    L1: for (j = 0; j < 80; j++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

      double temp2 = ((double)0);

      L2: for (k = 0; k < 60; k++) {
#pragma HLS unroll factor=2

        if (k < i) {
          C_buf[k][j] += alpha * B_buf[i][j] * A_buf[i][k];
          temp2 += B_buf[k][j] * A_buf[i][k];
        }
      }
      C_buf[i][j] = beta * C_buf[i][j] + alpha * B_buf[i][j] * A_buf[i][i] +
                    alpha * temp2;
    }
  }
  memcpy(C, C_buf, sizeof(double) * 60 * 80);
}
