#include "symm-opt-medium.h"

int main() {
  double alpha = 0.0;
  double beta = 0.0;
  double C[200][240];
  double A[200][200];
  double B[200][240];
  kernel_symm(alpha, beta, C, A, B);
  return 0;
}
