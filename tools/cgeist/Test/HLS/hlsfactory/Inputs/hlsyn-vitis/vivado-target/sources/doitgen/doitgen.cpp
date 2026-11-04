#include "doitgen.h"
void kernel_doitgen(int nr, int nq, int np, double A[25][20][30],
                    double C4[30][30], double sum[30]) {
  double A_buf[25][20][30];
  double C4_buf[30][30];
  double sum_buf[30];
  memcpy(A_buf, A, sizeof(double) * 25 * 20 * 30);
  memcpy(C4_buf, C4, sizeof(double) * 30 * 30);

  int r;
  int q;
  int p;
  int s;
  //#pragma scop

  L0: for (r = 0; r < 25; r++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    L1: for (q = 0; q < 20; q++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

      L2: for (p = 0; p < 30; p++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

        sum_buf[p] = 0.0;
        L3: for (s = 0; s < 30; s++) {
          sum_buf[p] += A_buf[r][q][s] * C4_buf[s][p];
        }
      }
      L4: for (p = 0; p < 30; p++) {
        A_buf[r][q][p] = sum_buf[p];
      }
    }
  }
  //#pragma endscop
  memcpy(A, A_buf, sizeof(double) * 25 * 20 * 30);
  memcpy(sum, sum_buf, sizeof(double) * 30);
}
