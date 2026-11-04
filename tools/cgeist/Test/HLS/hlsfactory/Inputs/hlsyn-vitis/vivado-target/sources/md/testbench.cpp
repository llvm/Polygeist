#include "md.h"

int main() {
  double force_x[256];
  double force_y[256];
  double force_z[256];
  double position_x[256];
  double position_y[256];
  double position_z[256];
  int NL[4096];
  md_kernel(force_x, force_y, force_z, position_x, position_y, position_z, NL);
  return 0;
}
