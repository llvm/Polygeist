#include <stdint.h>

static int32_t clamp_i32(int32_t x, int32_t lo, int32_t hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// Normalized uniform 3x3 window with round-to-nearest integer output.
void pva_fixture_box_filter_3x3_i8(int32_t h, int32_t w,
                                   const int8_t *restrict in,
                                   int8_t *restrict out) {
  for (int32_t y = 1; y < h - 1; ++y) {
    for (int32_t x = 1; x < w - 1; ++x) {
      int32_t sum = 0;
      for (int32_t ky = -1; ky <= 1; ++ky)
        for (int32_t kx = -1; kx <= 1; ++kx)
          sum += (int32_t)in[(y + ky) * w + x + kx];
      out[y * w + x] = (int8_t)clamp_i32((sum + 4) / 9, -128, 127);
    }
  }
}
