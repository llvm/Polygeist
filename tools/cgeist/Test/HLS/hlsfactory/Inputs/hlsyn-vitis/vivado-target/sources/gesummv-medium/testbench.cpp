#include "gesummv-medium.h"

int main() {
  double alpha = 0.0;
  double beta = 0.0;
  double A[250][250];
  double B[250][250];
  double tmp[250];
  double x[250];
  double y[250];
  kernel_gesummv(alpha, beta, A, B, tmp, x, y);
  return 0;
}
