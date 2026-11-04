#include "gemver-medium.h"
void kernel_gemver(int n, double alpha, double beta, double A[400][400],
                   double u1[400], double v1[400], double u2[400],
                   double v2[400], double w[400], double x[400], double y[400],
                   double z[400]) {
  double A_buf[400][400];
  double u1_buf[400];
  double v1_buf[400];
  double u2_buf[400];
  double v2_buf[400];
  double w_buf[400];
  double x_buf[400];
  double y_buf[400];
  double z_buf[400];
  memcpy(A_buf, A, sizeof(double) * 400 * 400);
  memcpy(u1_buf, u1, sizeof(double) * 400);
  memcpy(v1_buf, v1, sizeof(double) * 400);
  memcpy(u2_buf, u2, sizeof(double) * 400);
  memcpy(v2_buf, v2, sizeof(double) * 400);
  memcpy(w_buf, w, sizeof(double) * 400);
  memcpy(x_buf, x, sizeof(double) * 400);
  memcpy(y_buf, y, sizeof(double) * 400);
  memcpy(z_buf, z, sizeof(double) * 400);

  int i;
  int j;

  L0: for (i = 0; i < 400; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L1: for (j = 0; j < 400; j++) {
#pragma HLS unroll factor=2

      A_buf[i][j] += +u1_buf[i] * v1_buf[j] + u2_buf[i] * v2_buf[j];
    }
  }

  L2: for (i = 0; i < 400; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L3: for (j = 0; j < 400; j++) {
#pragma HLS unroll factor=2

      x_buf[i] += beta * A_buf[j][i] * y_buf[j];
    }
  }

  L4: for (i = 0; i < 400; i++) {
#pragma HLS unroll factor=2

    x_buf[i] = x_buf[i] + z_buf[i];
  }

  L5: for (i = 0; i < 400; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L6: for (j = 0; j < 400; j++) {
#pragma HLS unroll factor=2

      w_buf[i] += alpha * A_buf[i][j] * x_buf[j];
    }
  }
  memcpy(A, A_buf, sizeof(double) * 400 * 400);
  memcpy(w, w_buf, sizeof(double) * 400);
  memcpy(x, x_buf, sizeof(double) * 400);
}
