// Full-output correctness harness for a dtype-specific, structurally matched
// morphology dilation source function. The selected source function is linked
// from compiler-generated code; this harness never implements the replacement.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef PVA_MORPH_CTYPE
#error "define PVA_MORPH_CTYPE"
#endif
#ifndef PVA_MORPH_FUNCTION
#error "define PVA_MORPH_FUNCTION"
#endif
#ifndef PVA_MORPH_MIN
#error "define PVA_MORPH_MIN"
#endif
#ifndef PVA_MORPH_DTYPE
#error "define PVA_MORPH_DTYPE"
#endif

typedef PVA_MORPH_CTYPE pixel_t;
enum { H = 192, W = 256, PIXELS = H * W };
#define PVA_MORPH_STRINGIFY_INNER(value) #value
#define PVA_MORPH_STRINGIFY(value) PVA_MORPH_STRINGIFY_INNER(value)

void PVA_MORPH_FUNCTION(int, int, const pixel_t *, pixel_t *);

static uint64_t checksum(const pixel_t *data) {
  const uint8_t *bytes = (const uint8_t *)data;
  uint64_t value = 1469598103934665603ULL;
  for (size_t i = 0; i < sizeof(pixel_t) * PIXELS; ++i) {
    value ^= bytes[i];
    value *= 1099511628211ULL;
  }
  return value;
}

int main(void) {
  pixel_t *input = (pixel_t *)malloc(sizeof(pixel_t) * PIXELS);
  pixel_t *output = (pixel_t *)malloc(sizeof(pixel_t) * PIXELS);
  pixel_t *reference = (pixel_t *)malloc(sizeof(pixel_t) * PIXELS);
  if (!input || !output || !reference)
    return 2;

  // Exercise the full bit width, including negative values for signed types.
  for (size_t i = 0; i < PIXELS; ++i)
    input[i] = (pixel_t)(i * 40503u + i / 17u + 0x9e37u);
  memcpy(output, input, sizeof(pixel_t) * PIXELS);
  memcpy(reference, input, sizeof(pixel_t) * PIXELS);
  for (int y = 1; y < H - 1; ++y)
    for (int x = 1; x < W - 1; ++x) {
      pixel_t value = (pixel_t)PVA_MORPH_MIN;
      for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx) {
          pixel_t sample = input[(y + ky) * W + x + kx];
          if (sample > value)
            value = sample;
        }
      reference[y * W + x] = value;
    }

  PVA_MORPH_FUNCTION(H, W, input, output);
  size_t mismatches = 0;
  unsigned long long max_error = 0;
  for (size_t i = 0; i < PIXELS; ++i) {
    long long difference = (long long)output[i] - (long long)reference[i];
    unsigned long long error =
        (unsigned long long)(difference < 0 ? -difference : difference);
    mismatches += error != 0;
    if (error > max_error)
      max_error = error;
  }
  printf("MORPH_RAISED,%s,%s,mismatches=%zu,max_error=%llu,checksum=%llu\n",
         PVA_MORPH_STRINGIFY(PVA_MORPH_DTYPE),
         mismatches ? "FAIL" : "EXACT_PASS", mismatches,
         max_error, (unsigned long long)checksum(output));
  free(input);
  free(output);
  free(reference);
  return mismatches ? 1 : 0;
}
