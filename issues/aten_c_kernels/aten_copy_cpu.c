/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_copy_cpu(float input[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i] = input[i];
  }
#pragma endscop
}
