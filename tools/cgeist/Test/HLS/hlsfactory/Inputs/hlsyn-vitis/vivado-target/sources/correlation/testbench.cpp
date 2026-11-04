#include "correlation.h"

int main() {
  double float_n = 0.0;
  double data[100][80];
  double corr[80][80];
  double mean[80];
  double stddev[80];
  kernel_correlation(float_n, data, corr, mean, stddev);
  return 0;
}
