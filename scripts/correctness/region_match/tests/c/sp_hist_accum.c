/* Histogram accumulated into existing counts. */
void sp_hist_accum(int nsamples, const int *bin, int *counts) {
  for (int s = 0; s < nsamples; ++s) counts[bin[s]] = counts[bin[s]] + 1;
}
