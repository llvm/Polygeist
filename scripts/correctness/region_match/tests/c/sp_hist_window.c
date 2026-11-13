/* Histogram of samples [0, hi - lo) accumulated into counts (sample count is an expression). */
void sp_hist_window(int lo, int hi, const int *bin, int *counts) {
  for (int s = 0; s < hi - lo; ++s) counts[bin[s]] = counts[bin[s]] + 1;
}
