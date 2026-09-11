// Full-output harness for dtype-specific raised Box/Gaussian C operations.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef PVA_FILTER_CTYPE
#error "define PVA_FILTER_CTYPE"
#endif
#ifndef PVA_FILTER_FUNCTION
#error "define PVA_FILTER_FUNCTION"
#endif
#ifndef PVA_FILTER_DTYPE
#error "define PVA_FILTER_DTYPE"
#endif
#ifndef PVA_FILTER_KIND
#error "define PVA_FILTER_KIND: 1=box, 2=gaussian"
#endif
#ifndef PVA_FILTER_TOLERANCE
#define PVA_FILTER_TOLERANCE 0
#endif

typedef PVA_FILTER_CTYPE pixel_t;
enum { H = 192, W = 256, PIXELS = H * W };
#define STRINGIFY_INNER(value) #value
#define STRINGIFY(value) STRINGIFY_INNER(value)

void PVA_FILTER_FUNCTION(int, int, const pixel_t *, pixel_t *);

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
  pixel_t *input = malloc(sizeof(pixel_t) * PIXELS);
  pixel_t *output = malloc(sizeof(pixel_t) * PIXELS);
  pixel_t *reference = malloc(sizeof(pixel_t) * PIXELS);
  if (!input || !output || !reference) return 2;
  for (size_t i = 0; i < PIXELS; ++i)
    input[i] = (pixel_t)(i * 40503u + i / 17u + 0x9e37u);
  memcpy(output, input, sizeof(pixel_t) * PIXELS);
  memcpy(reference, input, sizeof(pixel_t) * PIXELS);
  for (int y = 1; y < H - 1; ++y)
    for (int x = 1; x < W - 1; ++x) {
      int sum = 0;
      for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx) {
#if PVA_FILTER_KIND == 1
          sum += input[(y + ky) * W + x + kx];
#elif PVA_FILTER_KIND == 2
          const int weight[3] = {1, 2, 1};
          sum += weight[ky + 1] * weight[kx + 1] *
                 input[(y + ky) * W + x + kx];
#else
#error "unsupported PVA_FILTER_KIND"
#endif
        }
#if PVA_FILTER_KIND == 1
      reference[y * W + x] = (pixel_t)((sum + 4) / 9);
#else
      reference[y * W + x] = (pixel_t)((sum + 8) >> 4);
#endif
    }
  PVA_FILTER_FUNCTION(H, W, input, output);
  size_t mismatches = 0;
  unsigned long long max_error = 0;
  for (size_t i = 0; i < PIXELS; ++i) {
    long long difference = (long long)output[i] - (long long)reference[i];
    unsigned long long error =
        (unsigned long long)(difference < 0 ? -difference : difference);
    mismatches += error != 0;
    if (error > max_error) max_error = error;
  }
  const char *status = max_error == 0 ? "EXACT_PASS" :
      max_error <= PVA_FILTER_TOLERANCE ? "WITHIN_TOLERANCE" : "FAIL";
  printf("FILTER_RAISED,%s,%s,%s,mismatches=%zu,max_error=%llu,tolerance=%d,checksum=%llu\n",
         PVA_FILTER_KIND == 1 ? "box" : "gaussian",
         STRINGIFY(PVA_FILTER_DTYPE), status, mismatches, max_error,
         PVA_FILTER_TOLERANCE, (unsigned long long)checksum(output));
  free(input); free(output); free(reference);
  return max_error <= PVA_FILTER_TOLERANCE ? 0 : 1;
}
