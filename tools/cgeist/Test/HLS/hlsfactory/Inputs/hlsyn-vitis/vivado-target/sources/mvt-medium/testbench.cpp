#include "mvt-medium.h"

int main() {
  double x1[400];
  double x2[400];
  double y_1[400];
  double y_2[400];
  double A[400][400];
  kernel_mvt(x1, x2, y_1, y_2, A);
  return 0;
}
