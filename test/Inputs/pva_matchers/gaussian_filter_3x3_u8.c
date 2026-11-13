#include <math.h>
#include <stdint.h>

static int32_t clamp_i32(int32_t x, int32_t lo, int32_t hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// Separable discrete Gaussian.  Keeping sigma and coefficient construction in
// the source lets a matcher prove the operation rather than guess from a
// familiar-looking constant stencil.
void pva_fixture_gaussian_filter_3x3_u8(int32_t h, int32_t w,
                                        const uint8_t *restrict in,
                                        uint8_t *restrict out, float sigma) {
  float one_d[3];
  float norm = 0.0f;
  for (int32_t k = -1; k <= 1; ++k) {
    float fk = (float)k;
    one_d[k + 1] = expf(-(fk * fk) / (2.0f * sigma * sigma));
    norm += one_d[k + 1];
  }
  for (int32_t k = 0; k < 3; ++k) one_d[k] /= norm;

  for (int32_t y = 1; y < h - 1; ++y) {
    for (int32_t x = 1; x < w - 1; ++x) {
      float sum = 0.0f;
      for (int32_t ky = 0; ky < 3; ++ky)
        for (int32_t kx = 0; kx < 3; ++kx)
          sum += one_d[ky] * one_d[kx] *
                 (float)in[(y + ky - 1) * w + x + kx - 1];
      out[y * w + x] = (uint8_t)clamp_i32((int32_t)(sum + 0.5f), 0, 255);
    }
  }
}
