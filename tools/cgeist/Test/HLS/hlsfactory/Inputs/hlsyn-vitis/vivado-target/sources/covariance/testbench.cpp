#include "covariance.h"

int main() {
  int m = 0;
  int n = 0;
  double float_n = 0.0;
  double data[100][80];
  double cov[80][80];
  double mean[80];
  kernel_covariance(m, n, float_n, data, cov, mean);
  return 0;
}
