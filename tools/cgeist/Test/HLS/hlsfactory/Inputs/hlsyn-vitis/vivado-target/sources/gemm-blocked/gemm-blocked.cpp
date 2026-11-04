#include "gemm-blocked.h"
void bbgemm(double m1[4096], double m2[4096], double prod[4096]) {
  double m1_buf[4096];
  double m2_buf[4096];
  double prod_buf[4096];
  memcpy(m1_buf, m1, sizeof(double) * 4096);
  memcpy(m2_buf, m2, sizeof(double) * 4096);
  memcpy(prod_buf, prod, sizeof(double) * 4096);

  int i;
  int k;
  int j;
  int jj;
  int kk;
  int i_row;
  int k_row;
  double temp_x;
  double mul;

  /* Standardize from: for(jj = 0;jj < 64;jj += 8) {...} */
  L0: for (jj = 0; jj <= 7; jj++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    int _in_jj = 0 + 8L * jj;

    /* Standardize from: for(kk = 0;kk < 64;kk += 8) {...} */
    L1: for (kk = 0; kk <= 7; kk++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

      int _in_kk = 0 + 8L * kk;

      L2: for (i = 0; i < 64; ++i) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

        L3: for (k = 0; k < 8; ++k) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

          i_row = i * 64;
          k_row = (k + _in_kk) * 64;
          temp_x = m1_buf[i_row + k + _in_kk];
          L4: for (j = 0; j < 8; ++j) {
            mul = temp_x * m2_buf[k_row + j + _in_jj];
            prod_buf[i_row + j + _in_jj] += mul;
          }
        }
      }
    }
    kk = 56 + 8L;
  }
  jj = 56 + 8L;
  memcpy(prod, prod_buf, sizeof(double) * 4096);
}
