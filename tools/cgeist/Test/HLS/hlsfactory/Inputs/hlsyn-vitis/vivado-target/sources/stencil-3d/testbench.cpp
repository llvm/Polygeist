#include "stencil-3d.h"

int main() {
  long C0 = 0;
  long C1 = 0;
  long orig[39304];
  long sol[32768];
  stencil3d(C0, C1, orig, sol);
  return 0;
}
