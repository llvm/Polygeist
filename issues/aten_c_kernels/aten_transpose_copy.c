/* Materializing aten::transpose/permute copy for a rank-2 tensor.
 * Upstream family: aten/src/ATen/native/TensorShape.cpp.
 */
#define M 32
#define N 24

void aten_transpose_copy(float input[M][N], float output[N][M]) {
#pragma scop
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      output[j][i] = input[i][j];
#pragma endscop
}
