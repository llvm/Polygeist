#include <math.h>
#include <stdio.h>
#include <string.h>

#define N 16
extern void kernel_conv2d(int ni, int nj, double *input, double *output);

int main(void) {
  double input[N][N], output[N][N];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      input[i][j] = (double)(i + j) / N;
  memset(output, 0, sizeof(output));
  kernel_conv2d(N, N, &input[0][0], &output[0][0]);
  double max_error = 0.0;
  for (int i = 1; i < N - 1; ++i) {
    for (int j = 1; j < N - 1; ++j) {
      double expected =
          0.2 * input[i-1][j-1] + 0.5 * input[i-1][j] -
          0.8 * input[i-1][j+1] - 0.3 * input[i][j-1] +
          0.6 * input[i][j] - 0.9 * input[i][j+1] +
          0.4 * input[i+1][j-1] + 0.7 * input[i+1][j] +
          0.1 * input[i+1][j+1];
      double error = fabs(output[i][j] - expected);
      if (error > max_error)
        max_error = error;
    }
  }
  if (max_error > 1.0e-12) {
    fprintf(stderr, "cuSten max error %.17g\n", max_error);
    return 1;
  }
  puts("cuSten 2D stencil: PASS");
  return 0;
}
