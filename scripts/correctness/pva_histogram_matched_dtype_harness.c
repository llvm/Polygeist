// Exact full-bin harness for raised ImageHistogram datatype/ABI routes.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef PVA_HIST_INPUT_CTYPE
#error "define PVA_HIST_INPUT_CTYPE"
#endif
#ifndef PVA_HIST_OUTPUT_CTYPE
#error "define PVA_HIST_OUTPUT_CTYPE"
#endif
#ifndef PVA_HIST_FUNCTION
#error "define PVA_HIST_FUNCTION"
#endif
#ifndef PVA_HIST_ROUTE
#error "define PVA_HIST_ROUTE"
#endif
#ifndef PVA_HIST_SHIFT
#error "define PVA_HIST_SHIFT"
#endif

typedef PVA_HIST_INPUT_CTYPE input_t;
typedef PVA_HIST_OUTPUT_CTYPE output_t;
enum { H = 192, W = 256, PIXELS = H * W };
#define STRINGIFY_INNER(value) #value
#define STRINGIFY(value) STRINGIFY_INNER(value)

void PVA_HIST_FUNCTION(int, int, const input_t *, output_t (*)[256]);

int main(void) {
  input_t *input = malloc(sizeof(input_t) * PIXELS);
  output_t output[256], reference[256] = {0};
  if (!input) return 2;
  for (size_t i = 0; i < PIXELS; ++i) {
    input[i] = (input_t)(i * 40503u + i / 17u + 0x9e37u);
    ++reference[((uint32_t)input[i]) >> PVA_HIST_SHIFT];
  }
  PVA_HIST_FUNCTION(H, W, input, &output);
  size_t mismatches = 0;
  unsigned long long max_error = 0, total_output = 0;
  for (int bin = 0; bin < 256; ++bin) {
    long long difference = (long long)output[bin] - reference[bin];
    unsigned long long error =
        (unsigned long long)(difference < 0 ? -difference : difference);
    mismatches += error != 0;
    if (error > max_error) max_error = error;
    total_output += (unsigned long long)output[bin];
  }
  printf("HIST_RAISED,%s,%s,mismatches=%zu,max_error=%llu,total=%llu\n",
         STRINGIFY(PVA_HIST_ROUTE), mismatches ? "FAIL" : "EXACT_PASS",
         mismatches, max_error, total_output);
  free(input);
  return mismatches ? 1 : 0;
}
