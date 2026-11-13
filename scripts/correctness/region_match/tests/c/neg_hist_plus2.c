/* NEG: adds 2 per sample, not a count. */
void neg_hist_plus2(int nsamples, const int *bin, int nbins, int *counts) {
  for (int b = 0; b < nbins; ++b) counts[b] = 0;
  for (int s = 0; s < nsamples; ++s) counts[bin[s]] += 2;
}
