/* Fixed-shape ATen linear 1D align_corners=false backward. */
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
void aten_upsample_linear1d_backward_cpu(float grad_output[B*C*O0], float grad_input[B*C*I0]) {
#pragma scop
  for (int p = 0; p < B*C*I0; ++p) grad_input[p] = 0.0f;
  for (int n = 0; n < B; ++n) {
    for (int c = 0; c < C; ++c) {
    for (int o0 = 0; o0 < O0; ++o0) {
      float s0 = ((float)o0 + 0.5f) * (float)I0 / (float)O0 - 0.5f;
      if (s0 < 0.0f) s0 = 0.0f;
      int i00 = (int)s0;
      int i01 = i00 + 1 < I0 ? i00 + 1 : i00;
      float w01 = s0 - (float)i00;
      float w00 = 1.0f - w01;
      grad_input[((((n)*C+c))*I0+i00)] += grad_output[((((n)*C+c))*O0+o0)] * w00;
      grad_input[((((n)*C+c))*I0+i01)] += grad_output[((((n)*C+c))*O0+o0)] * w01;
    }
  }
}
#pragma endscop
}
