// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 10
// clang-format on
/**
 * gemver.c: This file is part of the PolyBench 3.0 test suite.
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

#define RUN 50

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

__global__ void kernel_A(int n, double alpha, double beta, double *A,
                         double *u1, double *v1, double *u2, double *v2,
                         double *w, double *x, double *y, double *z) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < n && j < n)
    A[i * n + j] += u1[i] * v1[j] + u2[i] * v2[j];
}

__global__ void kernel_x(int n, double alpha, double beta, double *A,
                         double *u1, double *v1, double *u2, double *v2,
                         double *w, double *x, double *y, double *z) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    for (int j = 0; j < n; j++)
      x[i] += beta * A[j * n + i] * y[j];
  }
}

__global__ void kernel_y(int n, double alpha, double beta, double *A,
                         double *u1, double *v1, double *u2, double *v2,
                         double *w, double *x, double *y, double *z) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n)
    x[i] += z[i];
}

__global__ void kernel_w(int n, double alpha, double beta, double *A,
                         double *u1, double *v1, double *u2, double *v2,
                         double *w, double *x, double *y, double *z) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    for (int j = 0; j < n; j++)
      w[i] += alpha * A[i * n + j] * x[j];
  }
}

static void kernel(int n, double alpha, double beta, double *A, double *u1,
                   double *v1, double *u2, double *v2, double *w, double *x,
                   double *y, double *z) {

  const unsigned threadsPerBlock = 256;

  {
    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(n, block.x), num_blocks(n, block.y), 1);
    kernel_A<<<grid, block>>>(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
  }

  kernel_x<<<num_blocks(n, threadsPerBlock), threadsPerBlock>>>(
      n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
  kernel_y<<<num_blocks(n, threadsPerBlock), threadsPerBlock>>>(
      n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
  kernel_w<<<num_blocks(n, threadsPerBlock), threadsPerBlock>>>(
      n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
}

/* Array initialization. */
static void init_array(int n, double *A, double *u1, double *v1, double *u2,
                       double *v2, double *w, double *x, double *y, double *z) {
  int i, j;

  for (i = 0; i < n; i++) {
    u1[i] = i;
    u2[i] = (i + 1) / n / 2.0;
    v1[i] = (i + 1) / n / 4.0;
    v2[i] = (i + 1) / n / 6.0;
    y[i] = (i + 1) / n / 8.0;
    z[i] = (i + 1) / n / 9.0;
    x[i] = 0.0;
    w[i] = 0.0;
    for (j = 0; j < n; j++)
      A[i * n + j] = ((double)i * j) / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, double *w) {
  int i;

  for (i = 0; i < n; i++) {
    fprintf(stderr, "%0.2lf ", w[i]);
    if (i % 20 == 0)
      fprintf(stderr, "\n");
  }
}

int main(int argc, char **argv) {
  /* Retrieve problem size. */
  int n = atoi(argv[2]);
  int dump_code = atoi(argv[1]);

  /* Variable declaration/allocation. */
  double *A = (double *)malloc(n * n * sizeof(double));
  double *u1 = (double *)malloc(n * sizeof(double));
  double *v1 = (double *)malloc(n * sizeof(double));
  double *u2 = (double *)malloc(n * sizeof(double));
  double *v2 = (double *)malloc(n * sizeof(double));
  double *w = (double *)malloc(n * sizeof(double));
  double *x = (double *)malloc(n * sizeof(double));
  double *y = (double *)malloc(n * sizeof(double));
  double *z = (double *)malloc(n * sizeof(double));

  double alpha = 43532;
  double beta = 12313;
  /* Initialize array(s). */
  init_array(n, A, u1, v1, u2, v2, w, x, y, z);

  double *dev_A;
  double *dev_u1;
  double *dev_v1;
  double *dev_u2;
  double *dev_v2;
  double *dev_w;
  double *dev_x;
  double *dev_y;
  double *dev_z;
  double *dev_alpha;
  double *dev_beta;
  cudaMalloc(&dev_A, n * n * sizeof(double));
  cudaMalloc(&dev_u1, n * sizeof(double));
  cudaMalloc(&dev_v1, n * sizeof(double));
  cudaMalloc(&dev_u2, n * sizeof(double));
  cudaMalloc(&dev_v2, n * sizeof(double));
  cudaMalloc(&dev_w, n * sizeof(double));
  cudaMalloc(&dev_x, n * sizeof(double));
  cudaMalloc(&dev_y, n * sizeof(double));
  cudaMalloc(&dev_z, n * sizeof(double));
  cudaMalloc(&dev_alpha, sizeof(double));
  cudaMalloc(&dev_beta, sizeof(double));
  cudaMemcpy(dev_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_u1, u1, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_v1, v1, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_u2, u2, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_v2, v2, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_w, w, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, y, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_z, z, n * sizeof(double), cudaMemcpyHostToDevice);

  /* Run kernel. */
  kernel(n, alpha, beta, dev_A, dev_u1, dev_v1, dev_u2, dev_v2, dev_w, dev_x,
         dev_y, dev_z);
  cudaMemcpy(w, dev_w, n * sizeof(double), cudaMemcpyDeviceToHost);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (dump_code == 1)
    print_array(n, w);

  /* Be clean. */
  free((void *)A);
  free((void *)u1);
  free((void *)v1);
  free((void *)u2);
  free((void *)v2);
  free((void *)w);
  free((void *)x);
  free((void *)y);
  free((void *)z);

  return 0;
}
