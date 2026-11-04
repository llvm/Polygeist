#include "gemm-p-large.h"

int main() {
  int ni = 0;
  int nj = 0;
  int nk = 0;
  double alpha = 0.0;
  double beta = 0.0;
  double C[200][220];
  double A[200][240];
  double B[240][220];
  kernel_gemm(ni, nj, nk, alpha, beta, C, A, B);
  return 0;
}
