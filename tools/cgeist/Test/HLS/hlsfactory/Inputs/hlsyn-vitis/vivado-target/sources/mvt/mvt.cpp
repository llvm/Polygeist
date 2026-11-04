#include "mvt.h"
void kernel_mvt(double x1[120], double x2[120], double y_1[120],
                double y_2[120], double A[120][120]) {
  double x1_buf[120];
  double x2_buf[120];
  double y_1_buf[120];
  double y_2_buf[120];
  double A_buf[120][120];
  memcpy(x1_buf, x1, sizeof(double) * 120);
  memcpy(x2_buf, x2, sizeof(double) * 120);
  memcpy(y_1_buf, y_1, sizeof(double) * 120);
  memcpy(y_2_buf, y_2, sizeof(double) * 120);
  memcpy(A_buf, A, sizeof(double) * 120 * 120);

  int i;
  int j;
  //#pragma scop

  L0: for (i = 0; i < 120; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L1: for (j = 0; j < 120; j++) {
#pragma HLS unroll factor=2

      x1_buf[i] += A_buf[i][j] * y_1_buf[j];
    }
  }

  L2: for (i = 0; i < 120; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L3: for (j = 0; j < 120; j++) {
#pragma HLS unroll factor=2

      x2_buf[i] += A_buf[j][i] * y_2_buf[j];
    }
  }
  //#pragma endscop
  memcpy(x1, x1_buf, sizeof(double) * 120);
  memcpy(x2, x2_buf, sizeof(double) * 120);
}
