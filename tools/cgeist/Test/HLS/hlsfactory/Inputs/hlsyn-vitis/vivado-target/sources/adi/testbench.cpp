#include "adi.h"

int main() {
  int tsteps = 0;
  int n = 0;
  double u[60][60];
  double v[60][60];
  double p[60][60];
  double q[60][60];
  kernel_adi(tsteps, n, u, v, p, q);
  return 0;
}
