#include "symm-opt-medium.h"
void kernel_symm(double alpha, double beta, double C[200][240],
                 double A[200][200], double B[200][240]) {
  double C_buf[200][240];
  double A_buf[200][200];
  double B_buf[200][240];
  memcpy(C_buf, C, sizeof(double) * 200 * 240);
  memcpy(A_buf, A, sizeof(double) * 200 * 200);
  memcpy(B_buf, B, sizeof(double) * 200 * 240);

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

  L0: for (i = 0; i < 200; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    L1: for (j = 0; j < 240; j++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      double tmp = B_buf[i][j];

      L2: for (k = 0; k < 200; k++) {
#pragma HLS unroll factor=2

        if (k < i) {
          C_buf[k][j] += alpha * tmp * A_buf[i][k];
        }
      }
      double temp2 = (double)0;

      L3: for (k = 0; k < 200; k++) {
#pragma HLS unroll factor=2

        if (k < i) {
          temp2 += B_buf[k][j] * A_buf[i][k];
        }
      }
      C_buf[i][j] = beta * C_buf[i][j] + alpha * B_buf[i][j] * A_buf[i][i] +
                    alpha * temp2;
    }
  }
  memcpy(C, C_buf, sizeof(double) * 200 * 240);
}
