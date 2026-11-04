#include "trmm-opt.h"
void kernel_trmm(double alpha, double A[60][60], double B[60][80]) {
  double A_buf[60][60];
  double B_buf[60][80];
  memcpy(A_buf, A, sizeof(double) * 60 * 60);
  memcpy(B_buf, B, sizeof(double) * 60 * 80);

  // BLAS parameters
  // SIDE   = 'L'
  // UPLO   = 'L'
  // TRANSA = 'T'
  // DIAG   = 'U'
  //  => Form  B := alpha*A**T*B.
  //  A is MxM
  //  B is MxN

  L0: for (int i = 0; i < 60; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L1: for (int j = 0; j < 80; j++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      double sum = B_buf[i][j];
      L2: for (int k = 0; k < 60; k++) {
#pragma HLS unroll factor=2

        if (k > i) {
          sum += A_buf[k][i] * B_buf[k][j];
        }
      }
      B_buf[i][j] = alpha * sum;
    }
  }
  memcpy(B, B_buf, sizeof(double) * 60 * 80);
}
