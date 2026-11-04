#include "spmv-ellpack.h"

int main() {
  double nzval[4940];
  int cols[4940];
  double vec[494];
  double out[494];
  ellpack(nzval, cols, vec, out);
  return 0;
}
