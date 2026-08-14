/* aten::upsample_bilinear2d, aligned 4x4 to 8x8. Upstream: ATen/native/UpSampleBilinear2d.cpp. */
#ifndef B
#define B 2
#define C 3
#define H 4
#define W 4
#endif
void aten_upsample_bilinear2d(float input[B][C][H][W],
                              float output[B][C][2*H][2*W]) {
#pragma scop
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int oh = 0; oh < 2*H; ++oh)
        for (int ow = 0; ow < 2*W; ++ow) {
          int h0 = oh / 2;
          int w0 = ow / 2;
          int h1 = h0 + 1 < H ? h0 + 1 : h0;
          int w1 = w0 + 1 < W ? w0 + 1 : w0;
          float fh = (float)(oh % 2) * 0.5f;
          float fw = (float)(ow % 2) * 0.5f;
          output[b][c][oh][ow] =
              (1.0f-fh) * ((1.0f-fw)*input[b][c][h0][w0] +
                            fw*input[b][c][h0][w1]) +
              fh * ((1.0f-fw)*input[b][c][h1][w0] +
                    fw*input[b][c][h1][w1]);
        }
#pragma endscop
}
