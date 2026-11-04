#include "heat-3d.h"
void kernel_heat_3d(int tsteps, int n, double A[20][20][20],
                    double B[20][20][20]) {
  double A_buf[20][20][20];
  double B_buf[20][20][20];
  memcpy(A_buf, A, sizeof(double) * 20 * 20 * 20);
  memcpy(B_buf, B, sizeof(double) * 20 * 20 * 20);

  int t;
  int i;
  int j;
  int k;
  //#pragma scop

  L0: for (t = 1; t <= 40; t++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    L1: for (i = 1; i < 20 - 1; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

      L2: for (j = 1; j < 20 - 1; j++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

        L3: for (k = 1; k < 20 - 1; k++) {
          B_buf[i][j][k] = 0.125 * (A_buf[i + 1][j][k] - 2.0 * A_buf[i][j][k] +
                                    A_buf[i - 1][j][k]) +
                           0.125 * (A_buf[i][j + 1][k] - 2.0 * A_buf[i][j][k] +
                                    A_buf[i][j - 1][k]) +
                           0.125 * (A_buf[i][j][k + 1] - 2.0 * A_buf[i][j][k] +
                                    A_buf[i][j][k - 1]) +
                           A_buf[i][j][k];
        }
      }
    }

    L4: for (i = 1; i < 20 - 1; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

      L5: for (j = 1; j < 20 - 1; j++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

        L6: for (k = 1; k < 20 - 1; k++) {
          A_buf[i][j][k] = 0.125 * (B_buf[i + 1][j][k] - 2.0 * B_buf[i][j][k] +
                                    B_buf[i - 1][j][k]) +
                           0.125 * (B_buf[i][j + 1][k] - 2.0 * B_buf[i][j][k] +
                                    B_buf[i][j - 1][k]) +
                           0.125 * (B_buf[i][j][k + 1] - 2.0 * B_buf[i][j][k] +
                                    B_buf[i][j][k - 1]) +
                           B_buf[i][j][k];
        }
      }
    }
  }
  //#pragma endscop
  memcpy(A, A_buf, sizeof(double) * 20 * 20 * 20);
  memcpy(B, B_buf, sizeof(double) * 20 * 20 * 20);
}
