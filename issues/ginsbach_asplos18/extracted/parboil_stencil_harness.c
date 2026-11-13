#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void cpu_stencil(float c0, float c1, float *input, float *output,
                 int nx, int ny, int nz);

static int index3d(int nx, int ny, int i, int j, int k) {
  return i + nx * (j + ny * k);
}

int main(void) {
  const int nx = 4, ny = 4, nz = 4, count = nx * ny * nz;
  float *input = (float *)malloc((size_t)count * sizeof(float));
  float *output = (float *)calloc((size_t)count, sizeof(float));
  if (!input || !output) return 100;
  for (int i = 0; i < count; ++i) input[i] = (float)(i + 1);

  const float c0 = 2.0f, c1 = 0.5f;
  cpu_stencil(c0, c1, input, output, nx, ny, nz);
  for (int k = 1; k < nz - 1; ++k)
    for (int j = 1; j < ny - 1; ++j)
      for (int i = 1; i < nx - 1; ++i) {
        int p = index3d(nx, ny, i, j, k);
        float expected = c1 * (input[p + 1] + input[p - 1] +
                               input[p + nx] + input[p - nx] +
                               input[p + nx * ny] + input[p - nx * ny]) -
                         c0 * input[p];
        if (fabsf(output[p] - expected) > 1.0e-4f) {
          fprintf(stderr, "stencil mismatch at %d: got %g expected %g\n",
                  p, output[p], expected);
          return 1;
        }
      }
  free(input);
  free(output);
  puts("parboil-source-stencil: PASS");
  return 0;
}
