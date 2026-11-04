#include "gemm-blocked.h"

int main() {
  double m1[4096];
  double m2[4096];
  double prod[4096];
  bbgemm(m1, m2, prod);
  return 0;
}
