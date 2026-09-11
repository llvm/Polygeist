/* Fixed-shape ATen nearest-exact 1D backward. */
#ifndef B
#define B 1
#endif
#ifndef C
#define C 2
#endif
#ifndef I0
#define I0 4
#endif
#ifndef O0
#define O0 7
#endif
void aten_upsample_nearest_exact1d_backward_cpu(float grad_output[B*C*O0], float grad_input[B*C*I0]) {
#pragma scop
  for (int p = 0; p < B*C*I0; ++p) grad_input[p] = 0.0f;
  for (int n = 0; n < B; ++n) {
    for (int c = 0; c < C; ++c) {
    for (int o0 = 0; o0 < O0; ++o0) {
      int i0 = ((2 * o0 + 1) * I0) / (2 * O0);
      if (i0 >= I0) i0 = I0 - 1;
      grad_input[((((n)*C+c))*I0+i0)] += grad_output[((((n)*C+c))*O0+o0)];
    }
  }
}
#pragma endscop
}
