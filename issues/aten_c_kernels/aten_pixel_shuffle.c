/* aten::pixel_shuffle, upscale=2. Upstream: ATen/native/PixelShuffle.cpp. */
#ifndef B
#define B 2
#endif
#ifndef C
#define C 3
#endif
#ifndef R
#define R 2
#endif
#ifndef H
#define H 4
#endif
#ifndef W
#define W 4
#endif
void aten_pixel_shuffle(float input[B][C][R][R][H][W],
                        float output[B][C][H * R][W * R]) {
#pragma scop
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int rh = 0; rh < R; ++rh)
        for (int rw = 0; rw < R; ++rw)
          for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
              output[b][c][h * R + rh][w * R + rw] =
                  input[b][c][rh][rw][h][w];
#pragma endscop
}
