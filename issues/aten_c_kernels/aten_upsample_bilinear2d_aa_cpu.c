/* Fixed-shape ATen antialiased bilinear 2D. */
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
extern float sinf(float);
void aten_upsample_bilinear2d_aa_cpu(float input[B*C*I0*I1], float output[B*C*O0*O1]) {
  float sy_scale = (float)I0 / (float)O0;
  float sx_scale = (float)I1 / (float)O1;
  float fy_scale = sy_scale > 1.0f ? sy_scale : 1.0f;
  float fx_scale = sx_scale > 1.0f ? sx_scale : 1.0f;
  for (int n = 0; n < B; ++n) for (int c = 0; c < C; ++c)
  for (int oy = 0; oy < O0; ++oy) for (int ox = 0; ox < O1; ++ox) {
    float sy = ((float)oy + 0.5f) * sy_scale - 0.5f;
    float sx = ((float)ox + 0.5f) * sx_scale - 0.5f;
    float norm = 0.0f;
    float value = 0.0f;
    for (int iy = 0; iy < I0; ++iy) for (int ix = 0; ix < I1; ++ix) {
      float ay = sy - (float)iy; if (ay < 0.0f) ay = -ay; ay /= fy_scale;
      float ax = sx - (float)ix; if (ax < 0.0f) ax = -ax; ax /= fx_scale;
      float wy = ay < 1.0f ? (ay < 1.0f ? 1.0f - ay : 0.0f) : 0.0f;
      float wx = ax < 1.0f ? (ax < 1.0f ? 1.0f - ax : 0.0f) : 0.0f;
      norm += wy * wx;
    }
    for (int iy = 0; iy < I0; ++iy) for (int ix = 0; ix < I1; ++ix) {
      float ay = sy - (float)iy; if (ay < 0.0f) ay = -ay; ay /= fy_scale;
      float ax = sx - (float)ix; if (ax < 0.0f) ax = -ax; ax /= fx_scale;
      float wy = ay < 1.0f ? (ay < 1.0f ? 1.0f - ay : 0.0f) : 0.0f;
      float wx = ax < 1.0f ? (ax < 1.0f ? 1.0f - ax : 0.0f) : 0.0f;
        value += input[((n*C+c)*I0+iy)*I1+ix] * wy * wx;
    }
      output[((n*C+c)*O0+oy)*O1+ox] = value / norm;
  }
}
