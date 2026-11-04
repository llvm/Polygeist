#include "2mm.h"

int main() {
  int ni = 0;
  int nj = 0;
  int nk = 0;
  int nl = 0;
  double alpha = 0.0;
  double beta = 0.0;
  double tmp[40][50];
  double A[40][70];
  double B[70][50];
  double C[50][80];
  double D[40][80];
  kernel_2mm(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D);
  return 0;
}
