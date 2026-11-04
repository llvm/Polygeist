#include "3mm.h"
void kernel_3mm(int ni, int nj, int nk, int nl, int nm, double E[40][50],
                double A[40][60], double B[60][50], double F[50][70],
                double C[50][80], double D[80][70], double G[40][70]) {
  double E_buf[40][50];
  double A_buf[40][60];
  double B_buf[60][50];
  double F_buf[50][70];
  double C_buf[50][80];
  double D_buf[80][70];
  double G_buf[40][70];
  memcpy(A_buf, A, sizeof(double) * 40 * 60);
  memcpy(B_buf, B, sizeof(double) * 60 * 50);
  memcpy(C_buf, C, sizeof(double) * 50 * 80);
  memcpy(D_buf, D, sizeof(double) * 80 * 70);

  int i;
  int j;
  int k;
  //#pragma scop
  /* E := A*B */

  L0: for (i = 0; i < 40; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L1: for (j = 0; j < 50; j++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

      E_buf[i][j] = 0.0;
      L2: for (k = 0; k < 60; ++k) {
#pragma HLS unroll factor=2
        E_buf[i][j] += A_buf[i][k] * B_buf[k][j];
      }
    }
  }
  /* F := C*D */

  L3: for (i = 0; i < 50; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    L4: for (j = 0; j < 70; j++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      F_buf[i][j] = 0.0;

      L5: for (k = 0; k < 80; ++k) {
#pragma HLS unroll factor=2
        F_buf[i][j] += C_buf[i][k] * D_buf[k][j];
      }
    }
  }
  /* G := E*F */

  L6: for (i = 0; i < 40; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
    L7: for (j = 0; j < 70; j++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2
      G_buf[i][j] = 0.0;

      L8: for (k = 0; k < 50; ++k) {
#pragma HLS unroll factor=2
        G_buf[i][j] += E_buf[i][k] * F_buf[k][j];
      }
    }
  }
  //#pragma endscop
  memcpy(E, E_buf, sizeof(double) * 40 * 50);
  memcpy(F, F_buf, sizeof(double) * 50 * 70);
  memcpy(G, G_buf, sizeof(double) * 40 * 70);
}
