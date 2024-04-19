// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 10 10 10
// clang-format on
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void init_array(int n, double *A, double *B) {
  int i, j, k;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++) {
        A[(i * n + j) * n + k] = ((double)i + j + k) / n;
        B[(i * n + j) * n + k] = ((double)i + j + k + 1) / n;
      }
}

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

__global__ void kernel_stencil(int iter, double *A, double *B) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
  int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int k = blockDim.z * blockIdx.z + threadIdx.z + 1;

  if (i < iter - 1 && j < iter - 1 && k < iter - 1) {
    B[(i * iter + j) * iter + k] =
        (A[((i + 1) * iter + j) * iter + k] - 2 * A[(i * iter + j) * iter + k] +
         A[((i - 1) * iter + j) * iter + k]) /
            8 +
        (A[(i * iter + (j + 1)) * iter + k] - 2 * A[(i * iter + j) * iter + k] +
         A[(i * iter + (j - 1)) * iter + k]) /
            8 +
        (A[(i * iter + j) * iter + k + 1] - 2 * A[(i * iter + j) * iter + k] +
         A[(i * iter + j) * iter + k - 1]) /
            8 +
        A[(i * iter + j) * iter + k];
  }
}

static void kernel(int tsteps, int iter, double *A, double *B) {
  const unsigned int threadsPerBlock = 256;

  for (int t = 1; t <= tsteps; t++) {
    dim3 block(1, threadsPerBlock / 32, 32);
    dim3 grid(num_blocks(iter - 2, block.x), num_blocks(iter - 2, block.y),
              num_blocks(iter - 2, block.z));
    kernel_stencil<<<grid, block>>>(iter, A, B);
    kernel_stencil<<<grid, block>>>(iter, B, A);
  }
}

static void print_array(int n, double *A) {
  int i, j, k;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++) {
        fprintf(stderr, "%0.2lf ", A[(i * n + j) * n + k]);
        if (((i * n + j) * n + k) % 20 == 0)
          fprintf(stderr, "\n");
      }
  fprintf(stderr, "\n");
}

int main(int argc, char **argv) {

  int dump_code = atoi(argv[1]);
  int tsteps = atoi(argv[2]);
  int n = atoi(argv[3]);

  double *A = (double *)malloc(n * n * n * sizeof(double));
  double *B = (double *)malloc(n * n * n * sizeof(double));

  init_array(n, A, B);

  double *dev_A;
  double *dev_B;
  cudaMalloc(&dev_A, n * n * n * sizeof(double));
  cudaMalloc(&dev_B, n * n * n * sizeof(double));
  cudaMemcpy(dev_A, A, n * n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, n * n * n * sizeof(double), cudaMemcpyHostToDevice);

  kernel(tsteps, n, dev_A, dev_B);
  cudaMemcpy(A, dev_A, n * n * n * sizeof(double), cudaMemcpyDeviceToHost);

  if (dump_code == 1)
    print_array(n, A);

  free((void *)A);
  free((void *)B);
}
