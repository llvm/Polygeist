#include "jacobi-1d.h"

int main() {
  int tsteps = 0;
  int n = 0;
  double A[120];
  double B[120];
  kernel_jacobi_1d(tsteps, n, A, B);
  return 0;
}
