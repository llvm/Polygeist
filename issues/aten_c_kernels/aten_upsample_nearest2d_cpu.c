/* Fixed-shape ATen nearest 2D. */
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
#ifndef I1
#define I1 5
#endif
#ifndef O1
#define O1 8
#endif
void aten_upsample_nearest2d_cpu(float input[B*C*I0*I1], float output[B*C*O0*O1]) {
#pragma scop
  for (int n = 0; n < B; ++n) {
    for (int c = 0; c < C; ++c) {
    for (int o0 = 0; o0 < O0; ++o0) {
      for (int o1 = 0; o1 < O1; ++o1) {
        int i0 = (o0 * I0) / O0;
        if (i0 >= I0) i0 = I0 - 1;
        int i1 = (o1 * I1) / O1;
        if (i1 >= I1) i1 = I1 - 1;
        output[((((((n)*C+c))*O0+o0))*O1+o1)] = input[((((((n)*C+c))*I0+i0))*I1+i1)];
      }
    }
  }
}
#pragma endscop
}
