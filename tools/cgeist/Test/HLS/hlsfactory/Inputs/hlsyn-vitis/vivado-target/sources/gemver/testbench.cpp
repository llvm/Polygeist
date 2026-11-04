#include "gemver.h"

int main() {
  int n = 0;
  double alpha = 0.0;
  double beta = 0.0;
  double A[120][120];
  double u1[120];
  double v1[120];
  double u2[120];
  double v2[120];
  double w[120];
  double x[120];
  double y[120];
  double z[120];
  kernel_gemver(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
  return 0;
}
