#include "adi.h"
void kernel_adi(int tsteps, int n, double u[60][60], double v[60][60],
                double p[60][60], double q[60][60]) {
  double u_buf[60][60];
  double v_buf[60][60];
  double p_buf[60][60];
  double q_buf[60][60];
  memcpy(u_buf, u, sizeof(double) * 60 * 60);

  int t;
  int i;
  int j;
  double DX;
  double DY;
  double DT;
  double B1;
  double B2;
  double mul1;
  double mul2;
  double a;
  double b;
  double c;
  double d;
  double e;
  double f;
  //#pragma scop
  DX = 1.0 / ((double)60);
  DY = 1.0 / ((double)60);
  DT = 1.0 / ((double)40);
  B1 = 2.0;
  B2 = 1.0;
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);
  a = -mul1 / 2.0;
  b = 1.0 + mul1;
  c = a;
  d = -mul2 / 2.0;
  e = 1.0 + mul2;
  f = d;

  L0: for (t = 1; t <= 40; t++) {
    // Column Sweep
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    L1: for (i = 1; i < 59; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

      v_buf[0][i] = 1.0;
      p_buf[i][0] = 0.0;
      q_buf[i][0] = v_buf[0][i];

      L2: for (j = 1; j < 59; j++) {
#pragma HLS unroll factor=2

        p_buf[i][j] = -c / (a * p_buf[i][j - 1] + b);
        q_buf[i][j] = (-d * u_buf[j][i - 1] + (1.0 + 2.0 * d) * u_buf[j][i] -
                       f * u_buf[j][i + 1] - a * q_buf[i][j - 1]) /
                      (a * p_buf[i][j - 1] + b);
      }
      v_buf[60 - 1][i] = 1.0;
      /* Standardize from: for(j = 60 - 2;j >= 1;j--) {...} */

      L3: for (j = 0; j <= 57; j++) {
#pragma HLS unroll factor=2

        int _in_j_0 = 58 + -1 * j;
        v_buf[_in_j_0][i] =
            p_buf[i][_in_j_0] * v_buf[_in_j_0 + 1][i] + q_buf[i][_in_j_0];
      }
      j = 1 + -1;
    }
    // Row Sweep

    L4: for (i = 1; i < 59; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

      u_buf[i][0] = 1.0;
      p_buf[i][0] = 0.0;
      q_buf[i][0] = u_buf[i][0];

      L5: for (j = 1; j < 59; j++) {
#pragma HLS unroll factor=2

        p_buf[i][j] = -f / (d * p_buf[i][j - 1] + e);
        q_buf[i][j] = (-a * v_buf[i - 1][j] + (1.0 + 2.0 * a) * v_buf[i][j] -
                       c * v_buf[i + 1][j] - d * q_buf[i][j - 1]) /
                      (d * p_buf[i][j - 1] + e);
      }
      u_buf[i][60 - 1] = 1.0;
      /* Standardize from: for(j = 60 - 2;j >= 1;j--) {...} */

      L6: for (j = 0; j <= 57; j++) {
#pragma HLS unroll factor=2

        int _in_j = 58 + -1 * j;
        u_buf[i][_in_j] =
            p_buf[i][_in_j] * u_buf[i][_in_j + 1] + q_buf[i][_in_j];
      }
      j = 1 + -1;
    }
  }
  //#pragma endscop
  memcpy(u, u_buf, sizeof(double) * 60 * 60);
  memcpy(v, v_buf, sizeof(double) * 60 * 60);
  memcpy(p, p_buf, sizeof(double) * 60 * 60);
  memcpy(q, q_buf, sizeof(double) * 60 * 60);
}
