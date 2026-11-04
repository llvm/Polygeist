#include "atax-medium.h"
void kernel_atax(double A[390][410], double x[410], double y[410],
                 double tmp[390]) {
  double A_buf[390][410];
  double x_buf[410];
  double y_buf[410];
  double tmp_buf[390];
  memcpy(A_buf, A, sizeof(double) * 390 * 410);
  memcpy(x_buf, x, sizeof(double) * 410);

  int i;
  int j;
  L0: for (i = 0; i < 410; i++)
    y_buf[i] = ((double)0);

  L1: for (i = 0; i < 390; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    tmp_buf[i] = 0.0;

    L2: for (j = 0; j < 410; j++) {
#pragma HLS unroll factor=2

      tmp_buf[i] += A_buf[i][j] * x_buf[j];
    }

    L3: for (j = 0; j < 410; j++) {
#pragma HLS unroll factor=2

      y_buf[j] += A_buf[i][j] * tmp_buf[i];
    }
  }
  memcpy(y, y_buf, sizeof(double) * 410);
  memcpy(tmp, tmp_buf, sizeof(double) * 390);
}
