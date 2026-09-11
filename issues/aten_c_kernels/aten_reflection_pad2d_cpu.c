/* Fixed-shape ATen reflection padding 2D. */
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
#ifndef I1
#define I1 5
#endif
#ifndef P1
#define P1 2
#endif
#define O1 (I1+2*P1)
void aten_reflection_pad2d_cpu(float input[B*C*I0*I1], float output[B*C*O0*O1]) {
  for (int n=0; n<B; ++n) {
    for (int c=0; c<C; ++c) {
    for (int o0=0; o0<O0; ++o0) {
      for (int o1=0; o1<O1; ++o1) {
        int i0 = o0 - P0;
        if (i0 < 0) i0 = -i0;
        if (i0 >= I0) i0 = 2*I0-2-i0;
        int i1 = o1 - P1;
        if (i1 < 0) i1 = -i1;
        if (i1 >= I1) i1 = 2*I1-2-i1;
        output[((((((n)*C+c))*O0+o0))*O1+o1)] = input[((((((n)*C+c))*I0+i0))*I1+i1)];
      }
    }
  }
}
}
