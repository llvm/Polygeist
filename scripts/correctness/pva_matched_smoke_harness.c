/* Link/run harness for proving that a structurally selected PVA launch makes
 * it through the complete C -> MLIR -> matcher -> PVA ABI -> AArch64 path.
 * This file declares but never implements the selected source function.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef PVA_SMOKE_OP
#error "compile with -DPVA_SMOKE_OP=1..6"
#endif
#ifndef PVA_SMOKE_TOLERANCE
#define PVA_SMOKE_TOLERANCE 0
#endif

enum { H = 192, W = 256, PIXELS = H * W };

#if PVA_SMOKE_OP == 1
void fixture_box_filter3x3(int, int, const uint8_t *, uint8_t *);
#elif PVA_SMOKE_OP == 2
void fixture_gaussian_filter3x3(int, int, const uint8_t *, uint8_t *);
#elif PVA_SMOKE_OP == 3
void fixture_morphology_dilate3x3(int, int, const uint8_t *, uint8_t *);
#elif PVA_SMOKE_OP == 4
void fixture_bilateral_filter3x3(int, int, const uint8_t *, uint8_t *, float,
                                 float);
#elif PVA_SMOKE_OP == 5
void fixture_image_histogram_u8(int, int, const uint8_t *, uint32_t (*)[256]);
#elif PVA_SMOKE_OP == 6
void fixture_histogram_equalization(int, int, const uint8_t *, uint8_t *);
#else
#error "PVA_SMOKE_OP must be 1..6"
#endif

static uint64_t checksum_bytes(const uint8_t *data, size_t size) {
  uint64_t checksum = 1469598103934665603ULL;
  for (size_t i = 0; i < size; ++i) {
    checksum ^= data[i];
    checksum *= 1099511628211ULL;
  }
  return checksum;
}

static int clampi(int value, int lo, int hi) {
  return value < lo ? lo : value > hi ? hi : value;
}

static void source_reference(const uint8_t *input, uint8_t *reference) {
#if PVA_SMOKE_OP == 1 || PVA_SMOKE_OP == 2 || PVA_SMOKE_OP == 3 || PVA_SMOKE_OP == 4
  memcpy(reference, input, PIXELS);
  for (int y = 1; y < H - 1; ++y) for (int x = 1; x < W - 1; ++x) {
#if PVA_SMOKE_OP == 1
    int sum = 0;
    for (int ky = -1; ky <= 1; ++ky) for (int kx = -1; kx <= 1; ++kx)
      sum += input[(y + ky) * W + x + kx];
    reference[y * W + x] = (uint8_t)((sum + 4) / 9);
#elif PVA_SMOKE_OP == 2
    const int k[3] = {1, 2, 1}; int sum = 0;
    for (int ky = -1; ky <= 1; ++ky) for (int kx = -1; kx <= 1; ++kx)
      sum += k[ky + 1] * k[kx + 1] * input[(y + ky) * W + x + kx];
    reference[y * W + x] = (uint8_t)((sum + 8) >> 4);
#elif PVA_SMOKE_OP == 3
    uint8_t value = 0;
    for (int ky = -1; ky <= 1; ++ky) for (int kx = -1; kx <= 1; ++kx) {
      uint8_t sample = input[(y + ky) * W + x + kx];
      if (sample > value) value = sample;
    }
    reference[y * W + x] = value;
#elif PVA_SMOKE_OP == 4
    const float sigma_r = 25.0f, sigma_s = 1.5f;
    float ir = -0.5f / (sigma_r * sigma_r);
    float is = -0.5f / (sigma_s * sigma_s);
    float center = input[y * W + x], weights = 0.0f, values = 0.0f;
    for (int ky = -1; ky <= 1; ++ky) for (int kx = -1; kx <= 1; ++kx) {
      float sample = input[(y + ky) * W + x + kx], delta = sample - center;
      float weight = expf((float)(kx * kx + ky * ky) * is + delta * delta * ir);
      weights += weight; values += weight * sample;
    }
    reference[y * W + x] = (uint8_t)clampi((int)(values / weights + 0.5f), 0, 255);
#endif
  }
#elif PVA_SMOKE_OP == 6
  uint32_t histogram[256] = {0}, cdf[256], first = 0;
  for (int i = 0; i < PIXELS; ++i) ++histogram[input[i]];
  cdf[0] = histogram[0];
  for (int b = 1; b < 256; ++b) cdf[b] = cdf[b - 1] + histogram[b];
  for (int b = 0; b < 256; ++b) if (first == 0 && cdf[b]) first = cdf[b];
  uint32_t denominator = PIXELS > first ? PIXELS - first : 1;
  for (int i = 0; i < PIXELS; ++i) {
    int value = (int)((cdf[input[i]] - first) * 255 / denominator);
    reference[i] = (uint8_t)clampi(value, 0, 255);
  }
#else
  (void)input; (void)reference;
#endif
}

int main(void) {
  uint8_t *input = malloc(PIXELS);
  uint8_t *output = malloc(PIXELS);
  uint8_t *reference = malloc(PIXELS);
  if (!input || !output || !reference) return 2;
  for (size_t i = 0; i < PIXELS; ++i) {
    input[i] = (uint8_t)((i * 73 + i / 17 + 19) & 255);
    output[i] = input[i];
  }
  source_reference(input, reference);
#if PVA_SMOKE_OP == 1
  fixture_box_filter3x3(H, W, input, output);
#elif PVA_SMOKE_OP == 2
  fixture_gaussian_filter3x3(H, W, input, output);
#elif PVA_SMOKE_OP == 3
  fixture_morphology_dilate3x3(H, W, input, output);
#elif PVA_SMOKE_OP == 4
  fixture_bilateral_filter3x3(H, W, input, output, 25.0f, 1.5f);
#elif PVA_SMOKE_OP == 5
  uint32_t histogram[256];
  uint32_t expected[256] = {0};
  for (int i = 0; i < PIXELS; ++i) ++expected[input[i]];
  fixture_image_histogram_u8(H, W, input, &histogram);
  size_t histogram_mismatches = 0;
  for (int b = 0; b < 256; ++b)
    histogram_mismatches += histogram[b] != expected[b];
  printf("MATCHED_SMOKE,5,%s,mismatches=%zu,checksum=%llu\n",
         histogram_mismatches ? "FAIL" : "EXACT_PASS", histogram_mismatches,
         (unsigned long long)checksum_bytes((const uint8_t *)histogram,
                                            sizeof(histogram)));
  free(input);
  free(output);
  free(reference);
  return 0;
#elif PVA_SMOKE_OP == 6
  fixture_histogram_equalization(H, W, input, output);
#endif
  size_t mismatches = 0; int max_error = 0;
  for (int i = 0; i < PIXELS; ++i) {
    int error = abs((int)output[i] - (int)reference[i]);
    mismatches += error != 0;
    if (error > max_error) max_error = error;
  }
  const char *status = max_error == 0 ? "EXACT_PASS" :
                       max_error <= PVA_SMOKE_TOLERANCE ?
                       "WITHIN_TOLERANCE" : "FAIL";
  printf("MATCHED_SMOKE,%d,%s,mismatches=%zu,max_error=%d,tolerance=%d,checksum=%llu\n",
         PVA_SMOKE_OP, status, mismatches, max_error, PVA_SMOKE_TOLERANCE,
         (unsigned long long)checksum_bytes(output, PIXELS));
  free(input);
  free(output);
  free(reference);
  return max_error <= PVA_SMOKE_TOLERANCE ? 0 : 1;
}
