#ifndef N
#define N 4096
#endif
#ifndef M
#define M 257
#endif
void aten_isin_default_cpu(float elements[N], float test[M], int out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    int found = 0;
    for (int j = 0; j < M; ++j) found |= elements[i] == test[j];
    out[i] = found;
  }
#pragma endscop
}
