#include "atax.h"
void kernel_atax(int m, int n, double A[116][124], double x[124], double y[124],
                 double tmp[116]) {
  double A_buf[116][124];
  double x_buf[124];
  double y_buf[124];
  double tmp_buf[116];
  memcpy(A_buf, A, sizeof(double) * 116 * 124);
  memcpy(x_buf, x, sizeof(double) * 124);

  int i;
  int j;
  //#pragma scop
  L0: for (i = 0; i < 124; i++)
    y_buf[i] = ((double)0);

  L1: for (i = 0; i < 116; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    tmp_buf[i] = 0.0;

    L2: for (j = 0; j < 124; j++) {
#pragma HLS unroll factor=2

      tmp_buf[i] += A_buf[i][j] * x_buf[j];
    }

    L3: for (j = 0; j < 124; j++) {
#pragma HLS unroll factor=2

      y_buf[j] += A_buf[i][j] * tmp_buf[i];
    }
  }
  //#pragma endscop
  memcpy(y, y_buf, sizeof(double) * 124);
  memcpy(tmp, tmp_buf, sizeof(double) * 116);
}
