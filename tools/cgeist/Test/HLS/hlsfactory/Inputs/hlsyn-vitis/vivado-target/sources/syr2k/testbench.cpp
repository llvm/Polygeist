#include "syr2k.h"

int main() {
  double alpha = 0.0;
  double beta = 0.0;
  double C[80][80];
  double A[80][60];
  double B[80][60];
  kernel_syr2k(alpha, beta, C, A, B);
  return 0;
}
