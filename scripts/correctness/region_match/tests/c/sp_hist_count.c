/* CUB DeviceHistogram semantics: counts start at zero, one count per sample's bin. */
void sp_hist_count(int nsamples, const int *bin, int nbins, int *counts) {
  for (int b = 0; b < nbins; ++b) counts[b] = 0;
  for (int s = 0; s < nsamples; ++s) counts[bin[s]]++;
}
