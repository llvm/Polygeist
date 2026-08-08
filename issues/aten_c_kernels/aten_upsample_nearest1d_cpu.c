/* Fixed-shape ATen nearest 1D. */
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
void aten_upsample_nearest1d_cpu(float input[B*C*I0], float output[B*C*O0]) {
#pragma scop
  for (int n = 0; n < B; ++n) {
    for (int c = 0; c < C; ++c) {
    for (int o0 = 0; o0 < O0; ++o0) {
      int i0 = (o0 * I0) / O0;
      if (i0 >= I0) i0 = I0 - 1;
      output[((((n)*C+c))*O0+o0)] = input[((((n)*C+c))*I0+i0)];
    }
  }
}
#pragma endscop
}
