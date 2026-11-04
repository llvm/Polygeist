  #include "fdtd-2d.h"
void kernel_fdtd_2d(int tmax, int nx, int ny, double ex[60][80],
                    double ey[60][80], double hz[60][80], double _fict_[40]) {
  double ex_buf[60][80];
  double ey_buf[60][80];
  double hz_buf[60][80];
  double _fict__buf[40];
  memcpy(ex_buf, ex, sizeof(double) * 60 * 80);
  memcpy(hz_buf, hz, sizeof(double) * 60 * 80);
  memcpy(_fict__buf, _fict_, sizeof(double) * 40);

  int t;
  int i;
  int j;
  //#pragma scop

  L0: for (t = 0; t < 40; t++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L1: for (j = 0; j < 80; j++) {
#pragma HLS unroll factor=2

      ey_buf[0][j] = _fict__buf[t];
    }

    L2: for (i = 1; i < 60; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      L3: for (j = 0; j < 80; j++) {
#pragma HLS unroll factor=2

        ey_buf[i][j] = ey_buf[i][j] - 0.5 * (hz_buf[i][j] - hz_buf[i - 1][j]);
      }
    }

    L4: for (i = 0; i < 60; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      L5: for (j = 1; j < 80; j++) {
#pragma HLS unroll factor=2

        ex_buf[i][j] = ex_buf[i][j] - 0.5 * (hz_buf[i][j] - hz_buf[i][j - 1]);
      }
    }

    L6: for (i = 0; i < 59; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      L7: for (j = 0; j < 79; j++) {
#pragma HLS unroll factor=2

        hz_buf[i][j] = hz_buf[i][j] - 0.7 * (ex_buf[i][j + 1] - ex_buf[i][j] +
                                             ey_buf[i + 1][j] - ey_buf[i][j]);
      }
    }
  }
  //#pragma endscop
  memcpy(ex, ex_buf, sizeof(double) * 60 * 80);
  memcpy(ey, ey_buf, sizeof(double) * 60 * 80);
  memcpy(hz, hz_buf, sizeof(double) * 60 * 80);
}
