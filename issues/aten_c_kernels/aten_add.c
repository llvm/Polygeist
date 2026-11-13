/* In-place aten::add on an NCHW tensor: out += src. */
#ifndef B
#define B 2
#endif
#ifndef C
#define C 8
#endif
#ifndef H
#define H 16
#endif
#ifndef W
#define W 16
#endif

void aten_add(float src[B][C][H][W], float out[B][C][H][W]) {
#pragma scop
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
          out[b][c][h][w] += src[b][c][h][w];
#pragma endscop
}
