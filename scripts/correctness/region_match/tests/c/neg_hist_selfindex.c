/* NEG: later samples read bins modified by earlier iterations, so the counts
   are not a histogram of the input bins.  (The first version zeroed bin[s]
   after reading it; raising legally splits that into a histogram loop and a
   separate zero-fill, and the histogram part is then a correct match.) */
void neg_hist_selfindex(int nsamples, int *bin, int *counts) {
  for (int s = 0; s + 1 < nsamples; ++s) {
    counts[bin[s]] += 1;
    bin[s + 1] = bin[s];
  }
}
