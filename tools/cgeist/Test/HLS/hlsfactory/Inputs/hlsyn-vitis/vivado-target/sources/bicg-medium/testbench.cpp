#include "bicg-medium.h"

int main() {
  int m = 0;
  int n = 0;
  double A[410][390];
  double s[390];
  double q[410];
  double p[390];
  double r[410];
  kernel_bicg(m, n, A, s, q, p, r);
  return 0;
}
