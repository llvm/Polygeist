#include "gemver.h"
void kernel_gemver(int n, double alpha, double beta, double A[120][120],
                   double u1[120], double v1[120], double u2[120],
                   double v2[120], double w[120], double x[120], double y[120],
                   double z[120]) {
  double A_buf[120][120];
  double u1_buf[120];
  double v1_buf[120];
  double u2_buf[120];
  double v2_buf[120];
  double w_buf[120];
  double x_buf[120];
  double y_buf[120];
  double z_buf[120];
  memcpy(A_buf, A, sizeof(double) * 120 * 120);
  memcpy(u1_buf, u1, sizeof(double) * 120);
  memcpy(v1_buf, v1, sizeof(double) * 120);
  memcpy(u2_buf, u2, sizeof(double) * 120);
  memcpy(v2_buf, v2, sizeof(double) * 120);
  memcpy(w_buf, w, sizeof(double) * 120);
  memcpy(x_buf, x, sizeof(double) * 120);
  memcpy(y_buf, y, sizeof(double) * 120);
  memcpy(z_buf, z, sizeof(double) * 120);

  int i;
  int j;

#pragma scop

  L0: for (i = 0; i < 120; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L1: for (j = 0; j < 120; j++) {
#pragma HLS unroll factor=2

      A_buf[i][j] += u1_buf[i] * v1_buf[j] + u2_buf[i] * v2_buf[j];
    }
  }

  L2: for (i = 0; i < 120; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L3: for (j = 0; j < 120; j++) {
#pragma HLS unroll factor=2

      x_buf[i] += beta * A_buf[j][i] * y_buf[j];
    }
  }

  L4: for (i = 0; i < 120; i++) {
#pragma HLS unroll factor=2

    x_buf[i] += z_buf[i];
  }

  L5: for (i = 0; i < 120; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L6: for (j = 0; j < 120; j++) {
#pragma HLS unroll factor=2

      w_buf[i] += alpha * A_buf[i][j] * x_buf[j];
    }
  }

#pragma endscop
  memcpy(A, A_buf, sizeof(double) * 120 * 120);
  memcpy(w, w_buf, sizeof(double) * 120);
  memcpy(x, x_buf, sizeof(double) * 120);
}
