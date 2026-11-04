#include "covariance.h"
void kernel_covariance(int m, int n, double float_n, double data[100][80],
                       double cov[80][80], double mean[80]) {
  double data_buf[100][80];
  double cov_buf[80][80];
  double mean_buf[80];
  memcpy(data_buf, data, sizeof(double) * 100 * 80);

  int i;
  int j;
  int k;
  //#pragma scop

  L0: for (j = 0; j < 80; j++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    mean_buf[j] = 0.0;

    L1: for (i = 0; i < 100; i++) {
#pragma HLS unroll factor=2

      mean_buf[j] += data_buf[i][j];
    }
    mean_buf[j] /= float_n;
  }

  L2: for (i = 0; i < 100; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L3: for (j = 0; j < 80; j++) {
#pragma HLS unroll factor=2

      data_buf[i][j] -= mean_buf[j];
    }
  }

  L4: for (i = 0; i < 80; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L5: for (j = i; j < 80; j++) {
#pragma HLS pipeline 

      cov_buf[i][j] = 0.0;

      L6: for (k = 0; k < 100; k++) {
#pragma HLS unroll factor=2

        cov_buf[i][j] += data_buf[k][i] * data_buf[k][j];
      }
      cov_buf[i][j] /= float_n - 1.0;
      cov_buf[j][i] = cov_buf[i][j];
    }
  }
  //#pragma endscop
  memcpy(data, data_buf, sizeof(double) * 100 * 80);
  memcpy(cov, cov_buf, sizeof(double) * 80 * 80);
  memcpy(mean, mean_buf, sizeof(double) * 80);
}
