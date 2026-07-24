/* aten::max_pool2d, NCHW, 2x2 window and stride 2. */
#define B 2
#define C 8
#define H 16
#define W 16
#define K 2
#define S 2
#define OH ((H - K) / S + 1)
#define OW ((W - K) / S + 1)
#define NEG_INF (-3.402823466e38f)

void aten_max_pool2d(float input[B][C][H][W],
                     float output[B][C][OH][OW]) {
#pragma scop
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow)
          output[b][c][oh][ow] = NEG_INF;
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow)
          for (int kh = 0; kh < K; ++kh)
            for (int kw = 0; kw < K; ++kw) {
              float value = input[b][c][oh * S + kh][ow * S + kw];
              float current = output[b][c][oh][ow];
              output[b][c][oh][ow] = value > current ? value : current;
            }
#pragma endscop
}
