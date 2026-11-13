#include <stdint.h>

static int32_t clamp_i32(int32_t x, int32_t lo, int32_t hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// Textbook 256-bin histogram equalization expressed as four observable
// stages: histogram, inclusive CDF, normalized LUT, and indexed remap.
void pva_fixture_histogram_equalization_u8(int32_t h, int32_t w,
                                            const uint8_t *restrict in,
                                            uint8_t *restrict out) {
  int32_t histogram[256];
  int32_t cdf[256];
  uint8_t lut[256];
  for (int32_t b = 0; b < 256; ++b) histogram[b] = 0;
  for (int32_t y = 0; y < h; ++y)
    for (int32_t x = 0; x < w; ++x)
      histogram[(uint32_t)in[y * w + x]] += 1;

  cdf[0] = histogram[0];
  for (int32_t b = 1; b < 256; ++b) cdf[b] = cdf[b - 1] + histogram[b];
  int32_t cdf_min = 0;
  for (int32_t b = 0; b < 256; ++b)
    if (cdf_min == 0 && cdf[b] != 0) cdf_min = cdf[b];

  int32_t denominator = h * w - cdf_min;
  if (denominator < 1) denominator = 1;
  for (int32_t b = 0; b < 256; ++b) {
    const int32_t scaled = (cdf[b] - cdf_min) * 255 / denominator;
    lut[b] = (uint8_t)clamp_i32(scaled, 0, 255);
  }
  for (int32_t y = 0; y < h; ++y)
    for (int32_t x = 0; x < w; ++x)
      out[y * w + x] = lut[(uint32_t)in[y * w + x]];
}
