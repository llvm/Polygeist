#include "atax-medium.h"

int main() {
  double A[390][410];
  double x[410];
  double y[410];
  double tmp[390];
  kernel_atax(A, x, y, tmp);
  return 0;
}
