/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_nan_to_num(float x[N], float nan_value, float posinf_value, float neginf_value, float max_finite, float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = x[i] != x[i] ? nan_value : (x[i] > max_finite ? posinf_value : (x[i] < -max_finite ? neginf_value : x[i]));
  }
#pragma endscop
}
