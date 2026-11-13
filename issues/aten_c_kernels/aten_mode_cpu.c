#ifndef N
#define N 64
#endif
void aten_mode_cpu(float x[N], float out[1], int out_index[1]) {
  float work[N];
  int indices[N];
  for (int i = 0; i < N; ++i) { work[i] = x[i]; indices[i] = i; }
  for (int i = 1; i < N; ++i) {
    float v = work[i]; int idx = indices[i]; int j = i - 1;
    while (j >= 0 && work[j] > v) {
      work[j + 1] = work[j]; indices[j + 1] = indices[j]; --j;
    }
    work[j + 1] = v; indices[j + 1] = idx;
  }
  int best_count = 1, count = 1, best = 0;
  for (int i = 1; i < N; ++i) {
    if (work[i] == work[i - 1]) ++count; else count = 1;
    if (count > best_count) { best_count = count; best = i; }
  }
  out[0] = work[best]; out_index[0] = indices[best];
}
