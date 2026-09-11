/* Fixed-shape ATen nearest-exact 3D backward. */
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
#ifndef I2
#define I2 6
#endif
#ifndef O2
#define O2 9
#endif
void aten_upsample_nearest_exact3d_backward_cpu(float grad_output[B*C*O0*O1*O2], float grad_input[B*C*I0*I1*I2]) {
#pragma scop
  for (int p = 0; p < B*C*I0*I1*I2; ++p) grad_input[p] = 0.0f;
  for (int n = 0; n < B; ++n) {
    for (int c = 0; c < C; ++c) {
    for (int o0 = 0; o0 < O0; ++o0) {
      for (int o1 = 0; o1 < O1; ++o1) {
        for (int o2 = 0; o2 < O2; ++o2) {
          int i0 = ((2 * o0 + 1) * I0) / (2 * O0);
          if (i0 >= I0) i0 = I0 - 1;
          int i1 = ((2 * o1 + 1) * I1) / (2 * O1);
          if (i1 >= I1) i1 = I1 - 1;
          int i2 = ((2 * o2 + 1) * I2) / (2 * O2);
          if (i2 >= I2) i2 = I2 - 1;
          grad_input[((((((((n)*C+c))*I0+i0))*I1+i1))*I2+i2)] += grad_output[((((((((n)*C+c))*O0+o0))*O1+o1))*O2+o2)];
        }
      }
    }
  }
}
#pragma endscop
}
