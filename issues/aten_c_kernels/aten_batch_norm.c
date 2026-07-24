/* Inference aten::batch_norm with caller-provided reciprocal stddev. */
#define B 2
#define C 8
#define H 16
#define W 16

void aten_batch_norm(float input[B][C][H][W], float weight[C],
                     float mean[C], float inv_std[C], float bias[C],
                     float output[B][C][H][W]) {
#pragma scop
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
          output[b][c][h][w] =
              weight[c] * (input[b][c][h][w] - mean[c]) * inv_std[c] +
              bias[c];
#pragma endscop
}
