/* Fixed-shape ATen reflection padding 1D. */
#ifndef B
#define B 1
#endif
#ifndef C
#define C 2
#endif
#ifndef I0
#define I0 4
#endif
#ifndef P0
#define P0 2
#endif
#define O0 (I0+2*P0)
void aten_reflection_pad1d_cpu(float input[B*C*I0], float output[B*C*O0]) {
  for (int n=0; n<B; ++n) {
    for (int c=0; c<C; ++c) {
    for (int o0=0; o0<O0; ++o0) {
      int i0 = o0 - P0;
      if (i0 < 0) i0 = -i0;
      if (i0 >= I0) i0 = 2*I0-2-i0;
      output[((((n)*C+c))*O0+o0)] = input[((((n)*C+c))*I0+i0)];
    }
  }
}
}
