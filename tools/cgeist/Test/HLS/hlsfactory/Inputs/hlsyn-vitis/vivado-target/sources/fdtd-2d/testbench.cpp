#include "fdtd-2d.h"

int main() {
  int tmax = 0;
  int nx = 0;
  int ny = 0;
  double ex[60][80];
  double ey[60][80];
  double hz[60][80];
  double _fict_[40];
  kernel_fdtd_2d(tmax, nx, ny, ex, ey, hz, _fict_);
  return 0;
}
