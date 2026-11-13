#include <math.h>
#include <stdint.h>

static int32_t clamp_i32(int32_t x, int32_t lo, int32_t hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// Edge-preserving bilateral filter: a normalized product of spatial and
// intensity-distance Gaussian weights.
void pva_fixture_bilateral_filter_3x3_u8(
    int32_t h, int32_t w, const uint8_t *restrict in, uint8_t *restrict out,
    float sigma_range, float sigma_space) {
  const float inv_range = -0.5f / (sigma_range * sigma_range);
  const float inv_space = -0.5f / (sigma_space * sigma_space);
  for (int32_t y = 1; y < h - 1; ++y) {
    for (int32_t x = 1; x < w - 1; ++x) {
      const float center = (float)in[y * w + x];
      float weighted_sum = 0.0f;
      float weight_sum = 0.0f;
      for (int32_t ky = -1; ky <= 1; ++ky) {
        for (int32_t kx = -1; kx <= 1; ++kx) {
          const float sample = (float)in[(y + ky) * w + x + kx];
          const float delta = sample - center;
          const float distance2 = (float)(ky * ky + kx * kx);
          const float weight =
              expf(distance2 * inv_space + delta * delta * inv_range);
          weighted_sum += weight * sample;
          weight_sum += weight;
        }
      }
      const int32_t value = (int32_t)(weighted_sum / weight_sum + 0.5f);
      out[y * w + x] = (uint8_t)clamp_i32(value, 0, 255);
    }
  }
}
