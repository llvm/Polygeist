#include "gemver-medium.h"

int main() {
  int n = 0;
  double alpha = 0.0;
  double beta = 0.0;
  double A[400][400];
  double u1[400];
  double v1[400];
  double u2[400];
  double v2[400];
  double w[400];
  double x[400];
  double y[400];
  double z[400];
  kernel_gemver(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
  return 0;
}
