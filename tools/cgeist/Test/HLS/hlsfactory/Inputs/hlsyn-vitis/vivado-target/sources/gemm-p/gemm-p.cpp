#include "gemm-p.h"
void kernel_gemm(int ni, int nj, int nk, double alpha, double beta,
                 double C[60][70], double A[60][80], double B[80][70]) {
  double C_buf[60][70];
  double A_buf[60][80];
  double B_buf[80][70];
  memcpy(C_buf, C, sizeof(double) * 60 * 70);
  memcpy(A_buf, A, sizeof(double) * 60 * 80);
  memcpy(B_buf, B, sizeof(double) * 80 * 70);

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

#pragma scop

  L0: for (i = 0; i < 60; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L1: for (j = 0; j < 70; j++) {
#pragma HLS unroll factor=2

      C_buf[i][j] *= beta;
    }

    L2: for (k = 0; k < 80; k++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      L3: for (j = 0; j < 70; j++) {
#pragma HLS unroll factor=2

        C_buf[i][j] += alpha * A_buf[i][k] * B_buf[k][j];
      }
    }
  }

#pragma endscop
  memcpy(C, C_buf, sizeof(double) * 60 * 70);
}
