#include "gemm-ncubed.h"
void gemm(double m1[4096], double m2[4096], double prod[4096]) {
  double m1_buf[4096];
  double m2_buf[4096];
  double prod_buf[4096];
  memcpy(m1_buf, m1, sizeof(double) * 4096);
  memcpy(m2_buf, m2, sizeof(double) * 4096);

  int i;
  int j;
  int k;
  int k_col;
  int i_col;
  double mult;

  L0: for (i = 0; i < 64; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    L1: for (j = 0; j < 64; j++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

      i_col = i * 64;
      double sum = (double)0;

      L2: for (k = 0; k < 64; k++) {
#pragma HLS unroll factor=2
        k_col = k * 64;
        mult = m1_buf[i_col + k] * m2_buf[k_col + j];
        sum += mult;
      }
      prod_buf[i_col + j] = sum;
    }
  }
  memcpy(prod, prod_buf, sizeof(double) * 4096);
}
