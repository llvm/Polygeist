#include "gemm-p-large.h"
void kernel_gemm(int ni, int nj, int nk, double alpha, double beta,
                 double C[200][220], double A[200][240], double B[240][220]) {
  double C_buf[200][220];
  double A_buf[200][240];
  double B_buf[240][220];
  memcpy(C_buf, C, sizeof(double) * 200 * 220);
  memcpy(A_buf, A, sizeof(double) * 200 * 240);
  memcpy(B_buf, B, sizeof(double) * 240 * 220);

  int i;
  int j;
  int k;
  // BLAS PARAMS
  // TRANSA = 'N'
  // TRANSB = 'N'
  //  => Form C := alpha*A*B + beta*C,
  // A is NIxNK
  // B is NKxNJ
  // C is NIxNJ

  L0: for (i = 0; i < 200; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L1: for (j = 0; j < 220; j++) {
#pragma HLS unroll factor=2

      C_buf[i][j] *= beta;
    }

    L2: for (k = 0; k < 240; k++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      L3: for (j = 0; j < 220; j++) {
#pragma HLS unroll factor=2

        C_buf[i][j] += alpha * A_buf[i][k] * B_buf[k][j];
      }
    }
  }
  memcpy(C, C_buf, sizeof(double) * 200 * 220);
}
