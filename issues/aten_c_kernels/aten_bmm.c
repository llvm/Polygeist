/* aten::bmm numerical body.
 * Upstream family: aten/src/ATen/native/LinearAlgebra.cpp.
 */
#define BATCH 4
#define M 8
#define K 16
#define N 12

void aten_bmm(float a[BATCH][M][K], float b[BATCH][K][N],
              float out[BATCH][M][N]) {
#pragma scop
  for (int batch = 0; batch < BATCH; ++batch)
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)
        out[batch][i][j] = 0.0f;
  for (int batch = 0; batch < BATCH; ++batch)
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
          out[batch][i][j] += a[batch][i][k] * b[batch][k][j];
#pragma endscop
}
