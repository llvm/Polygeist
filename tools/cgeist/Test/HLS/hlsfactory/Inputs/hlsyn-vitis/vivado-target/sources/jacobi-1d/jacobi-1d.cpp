#include "jacobi-1d.h"
void kernel_jacobi_1d(int tsteps, int n, double A[120], double B[120]) {
  double A_buf[120];
  double B_buf[120];
  memcpy(A_buf, A, sizeof(double) * 120);

  int t;
  int i;
  //#pragma scop

  L0: for (t = 0; t < 40; t++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    L1: for (i = 1; i < 120 - 1; i++) {
#pragma HLS unroll factor=2

      B_buf[i] = 0.33333 * (A_buf[i - 1] + A_buf[i] + A_buf[i + 1]);
    }

    L2: for (i = 1; i < 120 - 1; i++) {
#pragma HLS unroll factor=2

      A_buf[i] = 0.33333 * (B_buf[i - 1] + B_buf[i] + B_buf[i + 1]);
    }
  }
  //#pragma endscop
  memcpy(A, A_buf, sizeof(double) * 120);
  memcpy(B, B_buf, sizeof(double) * 120);
}
