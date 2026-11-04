#include "spmv-crs.h"

int main() {
  double val[1666];
  int cols[1666];
  int rowDelimiters[495];
  double vec[494];
  double out[494];
  spmv(val, cols, rowDelimiters, vec, out);
  return 0;
}
