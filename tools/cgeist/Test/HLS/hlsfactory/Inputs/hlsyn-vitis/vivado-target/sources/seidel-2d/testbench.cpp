#include "seidel-2d.h"

int main() {
  int tsteps = 0;
  int n = 0;
  double A[120][120];
  kernel_seidel_2d(tsteps, n, A);
  return 0;
}
