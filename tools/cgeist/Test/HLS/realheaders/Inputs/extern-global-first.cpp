// A translation unit that only declares the global (and uses it), compiled
// before the one whose pragma names it.
#include <ap_fixed.h>
typedef ap_fixed<16, 5> T;
extern T W[2][10][100];
void xg_load(const T win[2][10][100]) {
  for (int l = 0; l < 2; l++)
    for (int o = 0; o < 10; o++)
      for (int i = 0; i < 100; i++) W[l][o][i] = win[l][o][i];
}
