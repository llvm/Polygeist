/* aten::adaptive_avg_pool3d, 8^3 to 4^3. Upstream: ATen/native/AdaptiveAveragePooling3d.cpp. */
#define B 2
#define C 3
#define D 8
#define H 8
#define W 8
void aten_adaptive_avg_pool3d(float input[B][C][D][H][W],
                              float output[B][C][4][4][4]) {
#pragma scop
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int od = 0; od < 4; ++od)
        for (int oh = 0; oh < 4; ++oh)
          for (int ow = 0; ow < 4; ++ow) {
            output[b][c][od][oh][ow] = 0.0f;
            for (int kd = 0; kd < 2; ++kd)
              for (int kh = 0; kh < 2; ++kh)
                for (int kw = 0; kw < 2; ++kw)
                  output[b][c][od][oh][ow] +=
                      input[b][c][2 * od + kd][2 * oh + kh][2 * ow + kw] /
                      8.0f;
          }
#pragma endscop
}
