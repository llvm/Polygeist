#include "stencil_stencil2d.h"

int main() {
  int orig[8192];
  int sol[8192];
  int filter[9];
  stencil(orig, sol, filter);
  return 0;
}
