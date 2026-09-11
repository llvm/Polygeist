/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_renorm_scale_factor(float norm[N], float maxnorm, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = norm[i] > maxnorm ? maxnorm / (norm[i] + 1.0e-7f) : 1.0f;
  }
#pragma endscop
}
