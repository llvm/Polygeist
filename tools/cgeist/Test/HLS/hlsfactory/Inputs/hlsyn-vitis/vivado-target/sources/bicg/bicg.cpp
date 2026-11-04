#include "bicg.h"
void kernel_bicg(int m, int n, double A[124][116], double s[116], double q[124],
                 double p[116], double r[124]) {
  double A_buf[124][116];
  double s_buf[116];
  double q_buf[124];
  double p_buf[116];
  double r_buf[124];
  memcpy(A_buf, A, sizeof(double) * 124 * 116);
  memcpy(p_buf, p, sizeof(double) * 116);
  memcpy(r_buf, r, sizeof(double) * 124);

  int i;
  int j;
  //#pragma scop

  L0: for (i = 0; i < 116; i++) {
#pragma HLS unroll factor=2

    s_buf[i] = ((double)0);
  }

  L1: for (i = 0; i < 124; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    q_buf[i] = 0.0;

    L2: for (j = 0; j < 116; j++) {
      s_buf[j] += r_buf[i] * A_buf[i][j];
      q_buf[i] += A_buf[i][j] * p_buf[j];
    }
  }
  //#pragma endscop
  memcpy(s, s_buf, sizeof(double) * 116);
  memcpy(q, q_buf, sizeof(double) * 124);
}
