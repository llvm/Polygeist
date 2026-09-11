/* aten::im2col, NCHW, 3x3 window, stride 1, no padding.
 * Upstream family: aten/src/ATen/native/Im2Col.cpp.
 */
#define B 2
#define C 3
#define H 8
#define W 8
#define KH 3
#define KW 3
#define OH (H - KH + 1)
#define OW (W - KW + 1)

void aten_im2col(float input[B][C][H][W],
                 float output[B][C][KH][KW][OH][OW]) {
#pragma scop
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw)
          for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow)
              output[b][c][kh][kw][oh][ow] =
                  input[b][c][oh + kh][ow + kw];
#pragma endscop
}
