#include "spmv-crs.h"
void spmv(double val[1666], int cols[1666], int rowDelimiters[495],
          double vec[494], double out[494]) {
  double val_buf[1666];
  int cols_buf[1666];
  int rowDelimiters_buf[495];
  double vec_buf[494];
  double out_buf[494];
  memcpy(val_buf, val, sizeof(double) * 1666);
  memcpy(cols_buf, cols, sizeof(int) * 1666);
  memcpy(rowDelimiters_buf, rowDelimiters, sizeof(int) * 495);
  memcpy(vec_buf, vec, sizeof(double) * 494);

  int i;
  int j;
  double sum;
  double Si;

  L0: for (i = 0; i < 494; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    sum = ((double)0);
    Si = ((double)0);
    int tmp_begin = rowDelimiters_buf[i];
    int tmp_end = rowDelimiters_buf[i + 1];
    L1: for (j = tmp_begin; j < tmp_end; j++) {
#pragma HLS loop_tripcount min = 1666 max = 1666 avg = 1666
      Si = val_buf[j] * vec_buf[cols_buf[j]];
      sum = sum + Si;
    }
    out_buf[i] = sum;
  }
  memcpy(out, out_buf, sizeof(double) * 494);
}
