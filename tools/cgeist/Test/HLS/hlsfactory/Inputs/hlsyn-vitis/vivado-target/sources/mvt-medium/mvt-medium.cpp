#include "mvt-medium.h"
void kernel_mvt(double x1[400], double x2[400], double y_1[400],
                double y_2[400], double A[400][400]) {
  double x1_buf[400];
  double x2_buf[400];
  double y_1_buf[400];
  double y_2_buf[400];
  double A_buf[400][400];
  memcpy(x1_buf, x1, sizeof(double) * 400);
  memcpy(x2_buf, x2, sizeof(double) * 400);
  memcpy(y_1_buf, y_1, sizeof(double) * 400);
  memcpy(y_2_buf, y_2, sizeof(double) * 400);
  memcpy(A_buf, A, sizeof(double) * 400 * 400);

  int i;
  int j;

  L0: for (i = 0; i < 400; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L1: for (j = 0; j < 400; j++) {
#pragma HLS unroll factor=2

      x1_buf[i] += A_buf[i][j] * y_1_buf[j];
    }
  }

  L2: for (i = 0; i < 400; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L3: for (j = 0; j < 400; j++) {
#pragma HLS unroll factor=2

      x2_buf[i] += A_buf[j][i] * y_2_buf[j];
    }
  }
  memcpy(x1, x1_buf, sizeof(double) * 400);
  memcpy(x2, x2_buf, sizeof(double) * 400);
}
