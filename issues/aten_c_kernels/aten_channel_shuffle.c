/* aten::channel_shuffle, groups=2. Upstream: ATen/native/ChanelShuffle.cpp. */
#define B 2
#define G 2
#define CPG 4
#define H 4
#define W 4
void aten_channel_shuffle(float input[B][G][CPG][H][W],
                          float output[B][CPG][G][H][W]) {
#pragma scop
  for (int b = 0; b < B; ++b)
    for (int g = 0; g < G; ++g)
      for (int c = 0; c < CPG; ++c)
        for (int h = 0; h < H; ++h)
          for (int w = 0; w < W; ++w)
            output[b][c][g][h][w] = input[b][g][c][h][w];
#pragma endscop
}
