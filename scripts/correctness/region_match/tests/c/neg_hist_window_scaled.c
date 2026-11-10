/* NEG: sample count hi - lo, but each sample adds 2 (not a bincount). */
void neg_hist_window_scaled(int lo, int hi, const int *bin, int *counts) {
  for (int s = 0; s < hi - lo; ++s) counts[bin[s]] = counts[bin[s]] + 2;
}
