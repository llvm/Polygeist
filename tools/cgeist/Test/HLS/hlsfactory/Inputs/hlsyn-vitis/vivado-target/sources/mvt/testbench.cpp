#include "mvt.h"

int main() {
  double x1[120];
  double x2[120];
  double y_1[120];
  double y_2[120];
  double A[120][120];
  kernel_mvt(x1, x2, y_1, y_2, A);
  return 0;
}
