/* Weighted histogram: per-bin sum of sample weights. */
void sp_hist_weighted(int nsamples, const int *bin, const double *w, int nbins, double *sum) {
  for (int b = 0; b < nbins; ++b) sum[b] = 0.0;
  for (int s = 0; s < nsamples; ++s) sum[bin[s]] += w[s];
}
