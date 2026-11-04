#include "stencil_stencil2d.h"
void stencil(int orig[8192], int sol[8192], int filter[9]) {
  int orig_buf[8192];
  int sol_buf[8192];
  int filter_buf[9];
  memcpy(orig_buf, orig, sizeof(int) * 8192);
  memcpy(filter_buf, filter, sizeof(int) * 9);

  int r;
  int c;
  int k1;
  int k2;
  int temp;
  int mul;

  L0: for (r = 0; r < 128 - 2; r++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    L1: for (c = 0; c < 64 - 2; c++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      temp = ((int)0);

      L2: for (k1 = 0; k1 < 3; k1++) {
#pragma HLS pipeline 
        L3: for (k2 = 0; k2 < 3; k2++) {
          mul = filter_buf[k1 * 3 + k2] * orig_buf[(r + k1) * 64 + c + k2];
          temp += mul;
        }
      }
      sol_buf[r * 64 + c] = temp;
    }
  }
  memcpy(sol, sol_buf, sizeof(int) * 8192);
}
