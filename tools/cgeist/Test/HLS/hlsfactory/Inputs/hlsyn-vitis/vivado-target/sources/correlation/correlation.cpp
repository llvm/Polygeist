#include "correlation.h"
void kernel_correlation(double float_n, double data[100][80],
                        double corr[80][80], double mean[80],
                        double stddev[80]) {
  double data_buf[100][80];
  double corr_buf[80][80];
  double mean_buf[80];
  double stddev_buf[80];
  memcpy(data_buf, data, sizeof(double) * 100 * 80);

  int i;
  int j;
  int k;
  double eps = 0.1;

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

  L2: for (j = 0; j < 80; j++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    stddev_buf[j] = 0.0;

    L3: for (i = 0; i < 100; i++) {
#pragma HLS unroll factor=2

      stddev_buf[j] += pow(data_buf[i][j] - mean_buf[j], (double)2);
    }
    stddev_buf[j] /= float_n;
    stddev_buf[j] = sqrt(stddev_buf[j]);
    /* The following in an inelegant but usual way to handle
             near-zero std. dev. values, which below would cause a zero-
             divide. */
    stddev_buf[j] = (stddev_buf[j] <= eps ? 1.0 : stddev_buf[j]);
  }
  /* Center and reduce the column vectors. */

  L4: for (i = 0; i < 100; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L5: for (j = 0; j < 80; j++) {
#pragma HLS unroll factor=2

      data_buf[i][j] -= mean_buf[j];
      data_buf[i][j] /= sqrt(float_n) * stddev_buf[j];
    }
  }
  /* Calculate the m * m correlation matrix. */

  L6: for (i = 0; i < 80 - 1; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    corr_buf[i][i] = 1.0;

    L7: for (j = i + 1; j < 80; j++) {
#pragma HLS pipeline 

      corr_buf[i][j] = 0.0;

      L8: for (k = 0; k < 100; k++) {
#pragma HLS unroll factor=2

        corr_buf[i][j] += data_buf[k][i] * data_buf[k][j];
      }
      corr_buf[j][i] = corr_buf[i][j];
    }
  }
  corr_buf[80 - 1][80 - 1] = 1.0;
  memcpy(data, data_buf, sizeof(double) * 100 * 80);
  memcpy(corr, corr_buf, sizeof(double) * 80 * 80);
  memcpy(mean, mean_buf, sizeof(double) * 80);
  memcpy(stddev, stddev_buf, sizeof(double) * 80);
}
