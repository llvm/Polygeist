// Diagnostic only: recover the effective U8 3x3 PVA box coefficients by
// observing the center output for one 255-valued impulse at each tap.
#include "polygeist_cublas_rt.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { H = 192, W = 256, PIXELS = H * W, CY = H / 2, CX = W / 2 };

int main(void) {
  static uint8_t input[PIXELS];
  static uint8_t output[PIXELS];
  puts("tap_y,tap_x,pva_output,source_round_sum_over_9");
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      memset(input, 0, sizeof(input));
      memset(output, 0, sizeof(output));
      input[(CY + dy) * W + CX + dx] = 255;
      polygeist_pva_boxfilter_3x3_u8(H, W, input, output);
      printf("%d,%d,%u,%u\n", dy, dx, output[CY * W + CX],
             (unsigned)((255 + 4) / 9));
    }
  }
  return 0;
}
