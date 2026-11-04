#include "stencil-3d.h"
void stencil3d(long C0, long C1, long orig[39304], long sol[32768]) {
  long orig_buf[39304];
  long sol_buf[32768];
  memcpy(orig_buf, orig, sizeof(long) * 39304);

  long sum0;
  long sum1;
  long mul0;
  long mul1;

  L0: for (long i = 1; i < 33; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    L1: for (long j = 1; j < 33; j++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

      L2: for (long ko = 0; ko <= 31; ko++) {
#pragma HLS unroll factor=2

        long _in_ko = 1L + 1L * ko;
        sum0 = orig_buf[_in_ko + (0) + (34) * (j + (34) * i)];
        sum1 = orig_buf[_in_ko + (0) + (34) * (j + (34) * (i + (1)))] +
               orig_buf[_in_ko + (0) + (34) * (j + (34) * (i - (1)))] +
               orig_buf[_in_ko + (0) + (34) * (j + (1) + (34) * i)] +
               orig_buf[_in_ko + (0) + (34) * (j - (1) + (34) * i)] +
               orig_buf[_in_ko + (0) + (1) + (34) * (j + (34) * i)] +
               orig_buf[_in_ko + (0) - (1) + (34) * (j + (34) * i)];
        mul0 = sum0 * C0;
        mul1 = sum1 * C1;
        sol_buf[_in_ko + (0) + (34) * (j + (34) * i)] = mul0 + mul1;
      }
    }
  }
  memcpy(sol, sol_buf, sizeof(long) * 32768);
}
