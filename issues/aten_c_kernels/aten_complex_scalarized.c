/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_complex_scalarized(float real[N], float imag[N], float out_re[N], float out_im[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out_re[i] = real[i];
    out_im[i] = imag[i];
  }
#pragma endscop
}
