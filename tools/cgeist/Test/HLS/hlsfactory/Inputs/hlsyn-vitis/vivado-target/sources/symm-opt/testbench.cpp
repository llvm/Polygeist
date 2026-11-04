#include "symm-opt.h"

int main() {
  double alpha = 0.0;
  double beta = 0.0;
  double C[60][80];
  double A[60][60];
  double B[60][80];
  kernel_symm(alpha, beta, C, A, B);
  return 0;
}
