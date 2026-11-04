#include "bicg.h"

int main() {
  int m = 0;
  int n = 0;
  double A[124][116];
  double s[116];
  double q[124];
  double p[116];
  double r[124];
  kernel_bicg(m, n, A, s, q, p, r);
  return 0;
}
