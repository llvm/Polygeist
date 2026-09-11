/* aten::upsample_nearest2d specialized to scale factor 2.
 * Upstream family: aten/src/ATen/native/UpSampleNearest2d.cpp.
 */
#define B 2
#define C 4
#define H 8
#define W 8
#define OH (2 * H)
#define OW (2 * W)

void aten_upsample_nearest2d(float input[B][C][H][W],
                             float output[B][C][OH][OW]) {
#pragma scop
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow)
          output[b][c][oh][ow] = input[b][c][oh / 2][ow / 2];
#pragma endscop
}
