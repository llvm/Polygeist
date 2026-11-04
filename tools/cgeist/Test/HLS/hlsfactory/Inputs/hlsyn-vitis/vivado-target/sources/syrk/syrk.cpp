#include "syrk.h"
void kernel_syrk(double alpha, double beta, double C[80][80],
                 double A[80][60]) {
  double C_buf[80][80];
  double A_buf[80][60];
  memcpy(C_buf, C, sizeof(double) * 80 * 80);
  memcpy(A_buf, A, sizeof(double) * 80 * 60);

  int i;
  int j;
  int k;
  // BLAS PARAMS
  // TRANS = 'N'
  // UPLO  = 'L'
  //  =>  Form  C := alpha*A*A**T + beta*C.
  // A is NxM
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
          C_buf[i][j] += alpha * A_buf[i][k] * A_buf[j][k];
        }
      }
    }
  }
  memcpy(C, C_buf, sizeof(double) * 80 * 80);
}
