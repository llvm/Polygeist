/* Fixed-shape ATen replication padding 1D_backward. */
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
void aten_replication_pad1d_backward_cpu(float grad_output[B*C*O0], float grad_input[B*C*I0]) {
  for (int p=0; p<B*C*I0; ++p) grad_input[p]=0.0f;
  for (int n=0; n<B; ++n) {
    for (int c=0; c<C; ++c) {
    for (int o0=0; o0<O0; ++o0) {
      int i0 = o0 - P0;
      if (i0 < 0) i0 = 0;
      if (i0 >= I0) i0 = I0-1;
      grad_input[((((n)*C+c))*I0+i0)] += grad_output[((((n)*C+c))*O0+o0)];
    }
  }
}
}
