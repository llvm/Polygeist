#include "doitgen-red.h"

int main() {
  double A[25][20][30];
  double C4[30][30];
  double sum[30];
  kernel_doitgen(A, C4, sum);
  return 0;
}
