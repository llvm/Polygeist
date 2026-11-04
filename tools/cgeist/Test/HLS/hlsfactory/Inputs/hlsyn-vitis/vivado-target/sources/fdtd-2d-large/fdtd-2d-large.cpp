#include "fdtd-2d-large.h"
void kernel_fdtd_2d(int tmax, int nx, int ny, double ex[200][240],
                    double ey[200][240], double hz[200][240],
                    double _fict_[100]) {
  double ex_buf[200][240];
  double ey_buf[200][240];
  double hz_buf[200][240];
  double _fict__buf[100];
  memcpy(ex_buf, ex, sizeof(double) * 200 * 240);
  memcpy(hz_buf, hz, sizeof(double) * 200 * 240);
  memcpy(_fict__buf, _fict_, sizeof(double) * 100);

  int t;
  int i;
  int j;

  L0: for (t = 0; t < 100; t++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L1: for (j = 0; j < 240; j++) {
#pragma HLS unroll factor=2

      ey_buf[0][j] = _fict__buf[t];
    }

    L2: for (i = 1; i < 200; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      L3: for (j = 0; j < 240; j++) {
#pragma HLS unroll factor=2

        ey_buf[i][j] = ey_buf[i][j] - 0.5 * (hz_buf[i][j] - hz_buf[i - 1][j]);
      }
    }

    L4: for (i = 0; i < 200; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      L5: for (j = 1; j < 240; j++) {
#pragma HLS unroll factor=2

        ex_buf[i][j] = ex_buf[i][j] - 0.5 * (hz_buf[i][j] - hz_buf[i][j - 1]);
      }
    }

    L6: for (i = 0; i < 200 - 1; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      L7: for (j = 0; j < 240 - 1; j++) {
#pragma HLS unroll factor=2

        hz_buf[i][j] = hz_buf[i][j] - 0.7 * (ex_buf[i][j + 1] - ex_buf[i][j] +
                                             ey_buf[i + 1][j] - ey_buf[i][j]);
      }
    }
  }
  memcpy(ex, ex_buf, sizeof(double) * 200 * 240);
  memcpy(ey, ey_buf, sizeof(double) * 200 * 240);
  memcpy(hz, hz_buf, sizeof(double) * 200 * 240);
}
