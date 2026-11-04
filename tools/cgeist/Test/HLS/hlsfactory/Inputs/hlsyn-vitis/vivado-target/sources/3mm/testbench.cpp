#include "3mm.h"

int main() {
  int ni = 0;
  int nj = 0;
  int nk = 0;
  int nl = 0;
  int nm = 0;
  double E[40][50];
  double A[40][60];
  double B[60][50];
  double F[50][70];
  double C[50][80];
  double D[80][70];
  double G[40][70];
  kernel_3mm(ni, nj, nk, nl, nm, E, A, B, F, C, D, G);
  return 0;
}
