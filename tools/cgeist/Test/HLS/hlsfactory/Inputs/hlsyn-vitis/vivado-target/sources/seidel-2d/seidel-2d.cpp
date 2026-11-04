#include "seidel-2d.h"
void kernel_seidel_2d(int tsteps, int n, double A[120][120]) {
  double A_buf[120][120];
  memcpy(A_buf, A, sizeof(double) * 120 * 120);

  int t;
  int i;
  int j;
  //#pragma scop

  L0: for (t = 0; t <= 39; t++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    L1: for (i = 1; i <= 118; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      L2: for (j = 1; j <= 118; j++) {
#pragma HLS unroll factor=2

        A_buf[i][j] =
            (A_buf[i - 1][j - 1] + A_buf[i - 1][j] + A_buf[i - 1][j + 1] +
             A_buf[i][j - 1] + A_buf[i][j] + A_buf[i][j + 1] +
             A_buf[i + 1][j - 1] + A_buf[i + 1][j] + A_buf[i + 1][j + 1]) /
            9.0;
      }
    }
  }
  //#pragma endscop
  memcpy(A, A_buf, sizeof(double) * 120 * 120);
}
