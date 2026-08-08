/* Fixed-shape ATen trilinear 3D align_corners=false. */
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
void aten_upsample_trilinear3d_cpu(float input[B*C*I0*I1*I2], float output[B*C*O0*O1*O2]) {
#pragma scop
  for (int n = 0; n < B; ++n) {
    for (int c = 0; c < C; ++c) {
    for (int o0 = 0; o0 < O0; ++o0) {
      for (int o1 = 0; o1 < O1; ++o1) {
        for (int o2 = 0; o2 < O2; ++o2) {
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
          float s2 = ((float)o2 + 0.5f) * (float)I2 / (float)O2 - 0.5f;
          if (s2 < 0.0f) s2 = 0.0f;
          int i20 = (int)s2;
          int i21 = i20 + 1 < I2 ? i20 + 1 : i20;
          float w21 = s2 - (float)i20;
          float w20 = 1.0f - w21;
          output[((((((((n)*C+c))*O0+o0))*O1+o1))*O2+o2)] = input[((((((((n)*C+c))*I0+i00))*I1+i10))*I2+i20)] * w00*w10*w20 + input[((((((((n)*C+c))*I0+i00))*I1+i10))*I2+i21)] * w00*w10*w21 + input[((((((((n)*C+c))*I0+i00))*I1+i11))*I2+i20)] * w00*w11*w20 + input[((((((((n)*C+c))*I0+i00))*I1+i11))*I2+i21)] * w00*w11*w21 + input[((((((((n)*C+c))*I0+i01))*I1+i10))*I2+i20)] * w01*w10*w20 + input[((((((((n)*C+c))*I0+i01))*I1+i10))*I2+i21)] * w01*w10*w21 + input[((((((((n)*C+c))*I0+i01))*I1+i11))*I2+i20)] * w01*w11*w20 + input[((((((((n)*C+c))*I0+i01))*I1+i11))*I2+i21)] * w01*w11*w21;
        }
      }
    }
  }
}
#pragma endscop
}
