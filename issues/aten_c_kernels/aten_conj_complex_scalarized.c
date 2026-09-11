/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_conj_complex_scalarized(float re[N], float im[N], float out_re[N], float out_im[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out_re[i] = re[i];
    out_im[i] = -im[i];
  }
#pragma endscop
}
