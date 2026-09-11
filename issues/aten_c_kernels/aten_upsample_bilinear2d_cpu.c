/* Fixed-shape ATen bilinear 2D align_corners=false. */
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
void aten_upsample_bilinear2d_cpu(float input[B*C*I0*I1], float output[B*C*O0*O1]) {
#pragma scop
  for (int n = 0; n < B; ++n) {
    for (int c = 0; c < C; ++c) {
    for (int o0 = 0; o0 < O0; ++o0) {
      for (int o1 = 0; o1 < O1; ++o1) {
        float s0 = ((float)o0 + 0.5f) * (float)I0 / (float)O0 - 0.5f;
        if (s0 < 0.0f) s0 = 0.0f;
        int i00 = (int)s0;
        int i01 = i00 + 1 < I0 ? i00 + 1 : i00;
        float w01 = s0 - (float)i00;
        float w00 = 1.0f - w01;
        float s1 = ((float)o1 + 0.5f) * (float)I1 / (float)O1 - 0.5f;
        if (s1 < 0.0f) s1 = 0.0f;
        int i10 = (int)s1;
        int i11 = i10 + 1 < I1 ? i10 + 1 : i10;
        float w11 = s1 - (float)i10;
        float w10 = 1.0f - w11;
        output[((((((n)*C+c))*O0+o0))*O1+o1)] = input[((((((n)*C+c))*I0+i00))*I1+i10)] * w00*w10 + input[((((((n)*C+c))*I0+i00))*I1+i11)] * w00*w11 + input[((((((n)*C+c))*I0+i01))*I1+i10)] * w01*w10 + input[((((((n)*C+c))*I0+i01))*I1+i11)] * w01*w11;
      }
    }
  }
}
#pragma endscop
}
