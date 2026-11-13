/* Fixed-shape ATen replication padding 3D_backward. */
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
#ifndef I2
#define I2 6
#endif
#ifndef P2
#define P2 2
#endif
#define O2 (I2+2*P2)
void aten_replication_pad3d_backward_cpu(float grad_output[B*C*O0*O1*O2], float grad_input[B*C*I0*I1*I2]) {
  for (int p=0; p<B*C*I0*I1*I2; ++p) grad_input[p]=0.0f;
  for (int n=0; n<B; ++n) {
    for (int c=0; c<C; ++c) {
    for (int o0=0; o0<O0; ++o0) {
      for (int o1=0; o1<O1; ++o1) {
        for (int o2=0; o2<O2; ++o2) {
          int i0 = o0 - P0;
          if (i0 < 0) i0 = 0;
          if (i0 >= I0) i0 = I0-1;
          int i1 = o1 - P1;
          if (i1 < 0) i1 = 0;
          if (i1 >= I1) i1 = I1-1;
          int i2 = o2 - P2;
          if (i2 < 0) i2 = 0;
          if (i2 >= I2) i2 = I2-1;
          grad_input[((((((((n)*C+c))*I0+i0))*I1+i1))*I2+i2)] += grad_output[((((((((n)*C+c))*O0+o0))*O1+o1))*O2+o2)];
        }
      }
    }
  }
}
}
