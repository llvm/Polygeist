#include "atax.h"

int main() {
  int m = 0;
  int n = 0;
  double A[116][124];
  double x[124];
  double y[124];
  double tmp[116];
  kernel_atax(m, n, A, x, y, tmp);
  return 0;
}
