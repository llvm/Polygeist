/* aten::adaptive_avg_pool2d specialized to 8x8 -> 4x4.
 * Upstream family: aten/src/ATen/native/AdaptiveAveragePooling.cpp.
 */
#define B 2
#define C 4
#define H 8
#define W 8
#define OH 4
#define OW 4

void aten_adaptive_avg_pool2d(float input[B][C][H][W],
                              float output[B][C][OH][OW]) {
#pragma scop
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow)
          output[b][c][oh][ow] = 0.0f;
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow)
          for (int kh = 0; kh < 2; ++kh)
            for (int kw = 0; kw < 2; ++kw)
              output[b][c][oh][ow] +=
                  0.25f * input[b][c][2 * oh + kh][2 * ow + kw];
#pragma endscop
}
