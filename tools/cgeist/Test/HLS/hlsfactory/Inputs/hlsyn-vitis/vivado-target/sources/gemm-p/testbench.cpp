#include "gemm-p.h"

int main() {
  int ni = 0;
  int nj = 0;
  int nk = 0;
  double alpha = 0.0;
  double beta = 0.0;
  double C[60][70];
  double A[60][80];
  double B[80][70];
  kernel_gemm(ni, nj, nk, alpha, beta, C, A, B);
  return 0;
}
