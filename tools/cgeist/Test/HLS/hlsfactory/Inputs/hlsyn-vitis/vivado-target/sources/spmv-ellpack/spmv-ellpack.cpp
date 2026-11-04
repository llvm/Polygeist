#include "spmv-ellpack.h"
void ellpack(double nzval[4940], int cols[4940], double vec[494],
             double out[494]) {
  double nzval_buf[4940];
  int cols_buf[4940];
  double vec_buf[494];
  double out_buf[494];
  memcpy(nzval_buf, nzval, sizeof(double) * 4940);
  memcpy(cols_buf, cols, sizeof(int) * 4940);
  memcpy(vec_buf, vec, sizeof(double) * 494);

  int i;
  int j;
  double Si;

  L0: for (i = 0; i < 494; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    double sum = out_buf[i];
    L1: for (j = 0; j < 10; j++) {
      Si = nzval_buf[j + i * 10] * vec_buf[cols_buf[j + i * 10]];
      sum += Si;
    }
    out_buf[i] = sum;
  }
  memcpy(out, out_buf, sizeof(double) * 494);
}
