#include "nw.h"

void needwun(char SEQA[128], char SEQB[128], char alignedA[256],
             char alignedB[256], int M[16641], char ptr[16641]) {
  int score;
  int up_left;
  int up;
  int left;
  int max;
  int row;
  int row_up;
  int r;
  int a_idx;
  int b_idx;
  int a_str_idx;
  int b_str_idx;

  L0: for (a_idx = 0; a_idx < 128 + 1; a_idx++) {
#pragma HLS unroll factor=2

    M[a_idx] = a_idx * -1;
  }

  L1: for (b_idx = 0; b_idx < 128 + 1; b_idx++) {
#pragma HLS unroll factor=2

    M[b_idx * (128 + 1)] = b_idx * -1;
  }
  // Matrix filling loop

  L2: for (b_idx = 1; b_idx < 128 + 1; b_idx++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    L3: for (a_idx = 1; a_idx < 128 + 1; a_idx++) {
#pragma HLS unroll factor=2

      if (((int)SEQA[a_idx - 1]) == ((int)SEQB[b_idx - 1])) {
        score = 1;
      } else {
        score = -1;
      }
      row_up = (b_idx - 1) * (128 + 1);
      row = b_idx * (128 + 1);
      up_left = M[row_up + (a_idx - 1)] + score;
      up = M[row_up + a_idx] + -1;
      left = M[row + (a_idx - 1)] + -1;
      max = (up_left > ((up > left ? up : left)) ? up_left
                                                 : ((up > left ? up : left)));
      M[row + a_idx] = max;
      if (max == left) {
        ptr[row + a_idx] = ((char)60);
      } else {
        if (max == up) {
          ptr[row + a_idx] = ((char)94);
        } else {
          ptr[row + a_idx] = ((char)92);
        }
      }
    }
  }
  // TraceBack (n.b. aligned sequences are backwards to avoid string appending)
  a_idx = 128;
  b_idx = 128;
  a_str_idx = 0;
  b_str_idx = 0;
  /*
      trace: while(a_idx>0 || b_idx>0) {
          r = b_idx*(ALEN+1);
          if (ptr[r + a_idx] == ALIGN){
              alignedA[a_str_idx++] = SEQA[a_idx-1];
              alignedB[b_str_idx++] = SEQB[b_idx-1];
              a_idx--;
              b_idx--;
          }
          else if (ptr[r + a_idx] == SKIPB){
              alignedA[a_str_idx++] = SEQA[a_idx-1];
              alignedB[b_str_idx++] = '-';
              a_idx--;
          }
          else{ // SKIPA
              alignedA[a_str_idx++] = '-';
              alignedB[b_str_idx++] = SEQB[b_idx-1];
              b_idx--;
          }
      }
      // Pad the result
      L4: for ( ; a_str_idx<ALEN+BLEN; a_str_idx++ ) {
        alignedA[a_str_idx] = '_';
      }
      L5: for ( ; b_str_idx<ALEN+BLEN; b_str_idx++ ) {
        alignedB[b_str_idx] = '_';
      }
  */
}
