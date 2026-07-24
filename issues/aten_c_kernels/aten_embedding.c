/* aten::embedding indexed gather. Upstream: ATen/native/Embedding.cpp. */
#define VOCAB 64
#define DIM 16
#define TOKENS 8
void aten_embedding(float weight[VOCAB][DIM], int indices[TOKENS],
                    float output[TOKENS][DIM]) {
#pragma scop
  for (int i = 0; i < TOKENS; ++i)
    for (int d = 0; d < DIM; ++d)
      output[i][d] = weight[indices[i]][d];
#pragma endscop
}
