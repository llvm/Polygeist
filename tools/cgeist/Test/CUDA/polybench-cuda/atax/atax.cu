// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 10 10
// clang-format on
/**
 * atax.c: This file is part of the PolyBench 3.0 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define RUN 100

__global__ void kernel3(int m, int n, double *A, double *x, double *y,
                        double *tmp) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < m) {
    double dot = 0.0;
    for (int j = 0; j < n; j++) {
      dot += A[i * n + j] * x[j];
    }
    tmp[i] = dot;
  }
}

__global__ void kernel4(int m, int n, double *A, double *x, double *y,
                        double *tmp) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < n) {
    double dot = 0;
    y[j] = 0;
    for (int i = 0; i < m; i++)
      dot += A[i * n + j] * tmp[i];
    y[j] = dot;
  }
}

static int num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

/* Array initialization. */
static void init_array(int nx, int ny, double *A, double *x, double *tmp,
                       double *y) {
  int i, j;

  for (i = 0; i < ny; i++) {
    x[i] = i * M_PI;
  }
  for (i = 0; i < nx; i++)
    tmp[i] = 0;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      A[i * ny + j] = ((double)i * (j + 1)) / nx;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nx, double *y)

{
  int i;

  for (i = 0; i < nx; i++) {
    fprintf(stderr, "%0.2lf ", y[i]);
    if (i % 20 == 0)
      fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

int main(int argc, char **argv) {
  /* Retrieve problem size. */
  int nx = atoi(argv[2]);
  int ny = atoi(argv[3]);
  int dump_code = atoi(argv[1]);

  /* Variable declaration/allocation. */
  double *A = (double *)malloc(nx * ny * sizeof(double));
  double *x = (double *)malloc(ny * sizeof(double));
  double *y = (double *)malloc(ny * sizeof(double));
  double *tmp = (double *)malloc(nx * sizeof(double));
  /* Initialize array(s). */
  init_array(nx, ny, A, x, tmp, y);

  double *dev_A;
  double *dev_x;
  double *dev_y;
  double *dev_tmp;
  cudaMalloc(&dev_A, nx * ny * sizeof(double));
  cudaMalloc(&dev_x, ny * sizeof(double));
  cudaMalloc(&dev_y, ny * sizeof(double));
  cudaMalloc(&dev_tmp, nx * sizeof(double));

  cudaMemcpy(dev_A, A, nx * ny * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_x, x, ny * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, y, ny * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_tmp, tmp, nx * sizeof(double), cudaMemcpyHostToDevice);

  const int threadsPerBlock = 256;
  kernel3<<<num_blocks(nx, threadsPerBlock), threadsPerBlock>>>(
      nx, ny, dev_A, dev_x, dev_y, dev_tmp);
  kernel4<<<num_blocks(ny, threadsPerBlock), threadsPerBlock>>>(
      nx, ny, dev_A, dev_x, dev_y, dev_tmp);

  cudaMemcpy(y, dev_y, ny * sizeof(double), cudaMemcpyDeviceToHost);
  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (dump_code == 1) {
    print_array(nx, y);
  }
  /* Be clean. */
  free((void *)A);
  free((void *)x);
  free((void *)y);
  free((void *)tmp);

  return 0;
}
