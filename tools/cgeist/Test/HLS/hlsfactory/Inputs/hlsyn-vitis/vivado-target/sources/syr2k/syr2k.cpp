#include "syr2k.h"
void kernel_syr2k(double alpha, double beta, double C[80][80], double A[80][60],
                  double B[80][60]) {
  double C_buf[80][80];
  double A_buf[80][60];
  double B_buf[80][60];
  memcpy(C_buf, C, sizeof(double) * 80 * 80);
  memcpy(A_buf, A, sizeof(double) * 80 * 60);
  memcpy(B_buf, B, sizeof(double) * 80 * 60);

  int i;
  int j;
  int k;
  // BLAS PARAMS
  // UPLO  = 'L'
  // TRANS = 'N'
  // A is NxM
  // B is NxM
  // C is NxN

  L0: for (i = 0; i < 80; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L1: for (j = 0; j < 80; j++) {
#pragma HLS unroll factor=2

      if (j <= i) {
        C_buf[i][j] *= beta;
      }
    }

    L2: for (k = 0; k < 60; k++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      L3: for (j = 0; j < 80; j++) {
#pragma HLS unroll factor=2

        if (j <= i) {
          C_buf[i][j] += A_buf[j][k] * alpha * B_buf[i][k] +
                         B_buf[j][k] * alpha * A_buf[i][k];
        }
      }
    }
  }
  memcpy(C, C_buf, sizeof(double) * 80 * 80);
}
