/* Materializing aten::transpose/permute copy for a rank-2 tensor.
 * Upstream family: aten/src/ATen/native/TensorShape.cpp.
 */
#ifndef M
#define M 32
#endif
#ifndef N
#define N 24
#endif

void aten_transpose_copy(float input[M][N], float output[N][M]) {
#pragma scop
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      output[j][i] = input[i][j];
#pragma endscop
}
