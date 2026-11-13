/* NEG: saturating 8-bit counter has no library equivalent. */
void neg_hist_saturate(int nsamples, const int *bin, unsigned char *counts) {
  for (int s = 0; s < nsamples; ++s)
    if (counts[bin[s]] < 255) counts[bin[s]]++;
}
