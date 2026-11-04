#include "doitgen.h"

int main() {
  int nr = 0;
  int nq = 0;
  int np = 0;
  double A[25][20][30];
  double C4[30][30];
  double sum[30];
  kernel_doitgen(nr, nq, np, A, C4, sum);
  return 0;
}
