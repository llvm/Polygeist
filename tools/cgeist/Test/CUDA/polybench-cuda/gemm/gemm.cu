// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 10 10 10
// RUN: cgeist %s %stdinclude %polymer_cudaopts -O3 -o %s.execm && %s.execm 1 10 10 10
// clang-format on
/**
 * gemm.c: This file is part of the PolyBench/C 3.2 test suite.
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

__global__ void kernel_dev(int ni, int nj, int nk, double alpha, double beta,
                           double *C, double *A, double *B) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k;
  double dot = 0.0;

  if (i < ni && j < nj) {
    dot = C[i * nj + j] * beta;
    for (k = 0; k < nk; k++)
      dot += alpha * A[i * nk + k] * B[k * nj + j];
    C[i * nj + j] = dot;
  }
}

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

static void kernel(int ni, int nj, int nk, double alpha, double beta, double *C,
                   double *A, double *B) {

  unsigned threadsPerBlock = 256;
  dim3 block(threadsPerBlock / 32, 32, 1);
  dim3 grid(num_blocks(ni, block.x), num_blocks(nj, block.y), 1);
  kernel_dev<<<grid, block>>>(ni, nj, nk, alpha, beta, C, A, B);
}

static void init_array(int ni, int nj, int nk, double *C, double *A,
                       double *B) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i * nj + j] = ((double)i * j) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i * nk + j] = ((double)i * j) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i * nj + j] = ((double)i * j) / ni;
}

static void print_array(int ni, int nj, double *C) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      fprintf(stderr, "%0.2lf ", C[i * nj + j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

int main(int argc, char **argv) {

  int dump_code = atoi(argv[1]);
  int ni = atoi(argv[2]);
  int nj = atoi(argv[3]);
  int nk = atoi(argv[4]);

  double *A = (double *)malloc(ni * nk * sizeof(double));
  double *B = (double *)malloc(nk * nj * sizeof(double));
  double *C = (double *)malloc(ni * nj * sizeof(double));

  double alpha = 32412;
  double beta = 2123;

  init_array(ni, nj, nk, C, A, B);

  double *dev_A;
  double *dev_B;
  double *dev_C;
  cudaMalloc(&dev_A, ni * nk * sizeof(double));
  cudaMalloc(&dev_B, nk * nj * sizeof(double));
  cudaMalloc(&dev_C, ni * nj * sizeof(double));
  cudaMemcpy(dev_A, A, ni * nk * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, nk * nj * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_C, C, ni * nj * sizeof(double), cudaMemcpyHostToDevice);

  kernel(ni, nj, nk, alpha, beta, dev_C, dev_A, dev_B);
  cudaMemcpy(C, dev_C, ni * nj * sizeof(double), cudaMemcpyDeviceToHost);

  if (dump_code == 1)
    print_array(ni, nj, C);

  free((void *)C);
  free((void *)A);
  free((void *)B);
  return 0;
}
