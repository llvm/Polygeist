// Source-faithful compute loop from Parboil histo/src/base/main.c.  File I/O,
// timing, and allocation remain outside the idiom in the original program.
void parboil_histo_core(int count, int num_bins, unsigned int img[count],
                         unsigned char histo[num_bins]) {
  for (unsigned int i = 0; i < (unsigned int)count; ++i) {
    const unsigned int value = img[i];
    if (histo[value] < 255)
      ++histo[value];
  }
}
