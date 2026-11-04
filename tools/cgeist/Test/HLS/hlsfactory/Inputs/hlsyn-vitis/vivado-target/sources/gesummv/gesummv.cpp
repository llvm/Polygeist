#include "gesummv.h"
void kernel_gesummv(int n, double alpha, double beta, double A[90][90],
                    double B[90][90], double tmp[90], double x[90],
                    double y[90]) {
  double A_buf[90][90];
  double B_buf[90][90];
  double tmp_buf[90];
  double x_buf[90];
  double y_buf[90];
  memcpy(A_buf, A, sizeof(double) * 90 * 90);
  memcpy(B_buf, B, sizeof(double) * 90 * 90);
  memcpy(x_buf, x, sizeof(double) * 90);

  int i;
  int j;

#pragma scop

  L0: for (i = 0; i < 90; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    tmp_buf[i] = 0.0;
    y_buf[i] = 0.0;

    L1: for (j = 0; j < 90; j++) {
#pragma HLS unroll factor=2

      tmp_buf[i] = A_buf[i][j] * x_buf[j] + tmp_buf[i];
      y_buf[i] = B_buf[i][j] * x_buf[j] + y_buf[i];
    }
    y_buf[i] = alpha * tmp_buf[i] + beta * y_buf[i];
  }

#pragma endscop
  memcpy(tmp, tmp_buf, sizeof(double) * 90);
  memcpy(y, y_buf, sizeof(double) * 90);
}
