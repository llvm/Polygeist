#include <stdio.h>

void parboil_histo_core(int count, int num_bins, unsigned int img[count],
                         unsigned char histo[num_bins]);

int main(void) {
  enum { bins = 37, count = 12000 };
  unsigned int image[count];
  unsigned char got[bins], expected[bins];
  for (int bin = 0; bin < bins; ++bin)
    got[bin] = expected[bin] = (unsigned char)((bin * 29) % 251);
  for (int i = 0; i < count; ++i) {
    image[i] = (unsigned int)((i * 17 + i / 11 + 3) % bins);
    unsigned int bin = image[i];
    if (expected[bin] < 255)
      ++expected[bin];
  }

  parboil_histo_core(count, bins, image, got);
  for (int bin = 0; bin < bins; ++bin) {
    if (got[bin] != expected[bin]) {
      fprintf(stderr,
              "parboil-histo-core: FAIL bin=%d got=%u expected=%u\n", bin,
              (unsigned)got[bin], (unsigned)expected[bin]);
      return 1;
    }
  }
  puts("parboil-histo-core: PASS");
  return 0;
}
