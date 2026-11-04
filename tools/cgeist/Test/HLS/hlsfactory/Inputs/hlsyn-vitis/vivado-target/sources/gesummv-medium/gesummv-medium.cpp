#include "gesummv-medium.h"
void kernel_gesummv(double alpha, double beta, double A[250][250],
                    double B[250][250], double tmp[250], double x[250],
                    double y[250]) {
  double A_buf[250][250];
  double B_buf[250][250];
  double tmp_buf[250];
  double x_buf[250];
  double y_buf[250];
  memcpy(A_buf, A, sizeof(double) * 250 * 250);
  memcpy(B_buf, B, sizeof(double) * 250 * 250);
  memcpy(x_buf, x, sizeof(double) * 250);

  int i;
  int j;

  L0: for (i = 0; i < 250; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    tmp_buf[i] = 0.0;
    y_buf[i] = 0.0;

    L1: for (j = 0; j < 250; j++) {
#pragma HLS unroll factor=2

      tmp_buf[i] += A_buf[i][j] * x_buf[j];
      y_buf[i] += B_buf[i][j] * x_buf[j];
    }
    y_buf[i] = alpha * tmp_buf[i] + beta * y_buf[i];
  }
  memcpy(tmp, tmp_buf, sizeof(double) * 250);
  memcpy(y, y_buf, sizeof(double) * 250);
}
