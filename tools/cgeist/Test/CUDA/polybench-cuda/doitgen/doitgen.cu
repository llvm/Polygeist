// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 10 10 10
// clang-format on
/**
 * doitgen.c: This file is part of the PolyBench/C 3.2 test suite.
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

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

__global__ void kernel_sum(int nr, int nq, int np, double *A, double *C4,
                           double *sum) {
  int r = blockDim.x * blockIdx.x + threadIdx.x;
  int q = blockDim.y * blockIdx.y + threadIdx.y;
  int p = blockDim.z * blockIdx.z + threadIdx.z;

  if (r < nr && q < nq && p < np) {
    double dot = 0.0;
    sum[(r * nq + q) * np + p] = 0;
    for (int s = 0; s < np; s++)
      dot += A[(r * nq + q) * np + s] * C4[s * np + p];
    sum[(r * nq + q) * np + p] = dot;
  }
}

static void kernel(int nr, int nq, int np, double *A, double *C4, double *sum) {

  const unsigned threadsPerBlock = 256;

  dim3 block(1, threadsPerBlock / 32, 32);
  dim3 grid(num_blocks(nr, block.x), num_blocks(nq, block.y),
            num_blocks(np, block.z));
  kernel_sum<<<grid, block>>>(nr, nq, np, A, C4, sum);
}

static void init_array(int nr, int nq, int np, double *A, double *C4) {
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
        A[i * np * nq + j * nq + k] = ((double)i * j + k) / np;
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      C4[i * np + j] = ((double)i * j) / np;
}

static void print_array(int nr, int nq, int np, double *A) {
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
        fprintf(stderr, "%0.2lf ", A[i * nq * np + j * nq + k]);
        if (i % 20 == 0)
          fprintf(stderr, "\n");
      }
  fprintf(stderr, "\n");
}

int main(int argc, char **argv) {

  int dump_code = atoi(argv[1]);
  int nr = atoi(argv[2]);
  int nq = atoi(argv[3]);
  int np = atoi(argv[4]);

  double *A = (double *)malloc(nr * nq * np * sizeof(double));
  double *sum = (double *)malloc(nr * nq * np * sizeof(double));
  double *C4 = (double *)malloc(np * np * sizeof(double));

  init_array(nr, nq, np, A, C4);

  double *dev_A;
  double *dev_sum;
  double *dev_C4;
  cudaMalloc(&dev_A, nr * nq * np * sizeof(double));
  cudaMalloc(&dev_sum, nr * nq * np * sizeof(double));
  cudaMalloc(&dev_C4, np * np * sizeof(double));
  cudaMemcpy(dev_A, A, nr * nq * np * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_C4, C4, np * np * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_sum, sum, nr * nq * np * sizeof(double),
             cudaMemcpyHostToDevice);

  kernel(nr, nq, np, dev_A, dev_C4, dev_sum);

  cudaMemcpy(sum, dev_sum, nr * nq * np * sizeof(double),
             cudaMemcpyDeviceToHost);

  if (dump_code == 1)
    print_array(nr, nq, np, sum);

  free((void *)A);
  ;
  free((void *)sum);
  ;
  free((void *)C4);
  ;

  return 0;
}
