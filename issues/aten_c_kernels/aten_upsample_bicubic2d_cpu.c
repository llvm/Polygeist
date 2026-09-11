/* Fixed-shape ATen bicubic 2D, align_corners=false, a=-0.75. */
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
static float aten_cubic_weight(float x) {
  if (x < 0.0f) x = -x;
  if (x < 1.0f) return ((1.25f * x - 2.25f) * x * x + 1.0f);
  if (x < 2.0f) return ((-0.75f * x + 3.75f) * x - 6.0f) * x + 3.0f;
  return 0.0f;
}
void aten_upsample_bicubic2d_cpu(float input[B*C*I0*I1], float output[B*C*O0*O1]) {
  for (int n = 0; n < B; ++n) for (int c = 0; c < C; ++c)
  for (int oy = 0; oy < O0; ++oy) for (int ox = 0; ox < O1; ++ox) {
    float sy = ((float)oy + 0.5f) * (float)I0 / (float)O0 - 0.5f;
    float sx = ((float)ox + 0.5f) * (float)I1 / (float)O1 - 0.5f;
    int by = (int)sy, bx = (int)sx;
    if (sy < 0.0f && sy != (float)by) --by;
    if (sx < 0.0f && sx != (float)bx) --bx;
    float value = 0.0f;
    for (int ky = -1; ky <= 2; ++ky) for (int kx = -1; kx <= 2; ++kx) {
      int iy = by + ky; if (iy < 0) iy = 0; if (iy >= I0) iy = I0 - 1;
      int ix = bx + kx; if (ix < 0) ix = 0; if (ix >= I1) ix = I1 - 1;
      value += input[((n*C+c)*I0+iy)*I1+ix] *
          aten_cubic_weight(sy-(float)(by+ky)) *
          aten_cubic_weight(sx-(float)(bx+kx));
    }
    output[((n*C+c)*O0+oy)*O1+ox] = value;
  }
}
