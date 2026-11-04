#include "gesummv.h"

int main() {
  int n = 0;
  double alpha = 0.0;
  double beta = 0.0;
  double A[90][90];
  double B[90][90];
  double tmp[90];
  double x[90];
  double y[90];
  kernel_gesummv(n, alpha, beta, A, B, tmp, x, y);
  return 0;
}
