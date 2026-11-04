#include "syrk.h"

int main() {
  double alpha = 0.0;
  double beta = 0.0;
  double C[80][80];
  double A[80][60];
  kernel_syrk(alpha, beta, C, A);
  return 0;
}
