#include <stdint.h>

static int32_t clamp_i32(int32_t x, int32_t lo, int32_t hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// PVA-style signed image convolution: unsigned Q8 coefficients, rounded and
// saturated to i8.  Only the valid interior is written.
void pva_fixture_conv3x3_i8(int32_t h, int32_t w, const int8_t *restrict in,
                            int8_t *restrict out,
                            const uint8_t *restrict coeff) {
  for (int32_t y = 1; y < h - 1; ++y) {
    for (int32_t x = 1; x < w - 1; ++x) {
      int32_t sum = 0;
      for (int32_t ky = 0; ky < 3; ++ky)
        for (int32_t kx = 0; kx < 3; ++kx)
          sum += (int32_t)in[(y + ky - 1) * w + x + kx - 1] *
                 (int32_t)coeff[ky * 3 + kx];
      out[y * w + x] = (int8_t)clamp_i32((sum + 128) >> 8, -128, 127);
    }
  }
}
