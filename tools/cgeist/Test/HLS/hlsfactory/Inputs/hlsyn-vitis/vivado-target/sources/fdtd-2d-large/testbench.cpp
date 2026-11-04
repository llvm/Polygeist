#include "fdtd-2d-large.h"

int main() {
  int tmax = 0;
  int nx = 0;
  int ny = 0;
  double ex[200][240];
  double ey[200][240];
  double hz[200][240];
  double _fict_[100];
  kernel_fdtd_2d(tmax, nx, ny, ex, ey, hz, _fict_);
  return 0;
}
