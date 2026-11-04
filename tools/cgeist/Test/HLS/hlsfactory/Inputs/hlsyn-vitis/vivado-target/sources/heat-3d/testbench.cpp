#include "heat-3d.h"

int main() {
  int tsteps = 0;
  int n = 0;
  double A[20][20][20];
  double B[20][20][20];
  kernel_heat_3d(tsteps, n, A, B);
  return 0;
}
