// clang-format off
// XFAIL: *
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm
// clang-format on
/**
 * 3mm.c: This file is part of the PolyBench/C 3.2 test suite.
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

__global__ void kernel_A_mul_B(int ni, int nj, int nk, double *C, double *A,
                               double *B) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  double dot = 0.0;

  if (i < ni && j < nj) {
    for (int k = 0; k < nk; k++)
      // C[i * nj + j] += A[i * nk + k] * B[k * nj + j];
      dot += A[i * nk + k] * B[k * nj + j];
    C[i * nj + j] = dot;
  }
}

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

static void init_array(int ni, int nj, int nk, int nl, int nm, double *A,
                       double *B, double *C, double *D, double *E, double *F,
                       double *G) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i * ni + j] = ((double)i * j) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i * nk + j] = ((double)i * (j + 1)) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i * nj + j] = ((double)i * (j + 3)) / nl;
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i * nm + j] = ((double)i * (j + 2)) / nk;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      E[i * ni + j] = 0;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      F[i * nj + j] = 0;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      G[i * ni + j] = 0;
}

static void print_array(int ni, int nl, double *G) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      fprintf(stderr, "%0.2lf ", G[i * ni + j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

static void kernel(int ni, int nj, int nk, int nl, int nm, double *E, double *A,
                   double *B, double *F, double *C, double *D, double *G) {
  unsigned threadsPerBlock = 256;
  dim3 block(threadsPerBlock / 32, 32, 1);

  {
    dim3 grid(num_blocks(ni, block.x), num_blocks(nj, block.y), 1);
    kernel_A_mul_B<<<grid, block>>>(ni, nj, nk, E, A, B);
  }

  {
    dim3 grid(num_blocks(nj, block.x), num_blocks(nl, block.y), 1);
    kernel_A_mul_B<<<grid, block>>>(nj, nl, nm, F, C, D);
  }

  {
    dim3 grid(num_blocks(ni, block.x), num_blocks(nl, block.y), 1);
    kernel_A_mul_B<<<grid, block>>>(ni, nl, nj, G, E, F);
  }
}

int main(int argc, char **argv) {

  int dump_code = atoi(argv[1]);
  int ni = atoi(argv[2]);
  int nj = atoi(argv[3]);
  int nk = atoi(argv[4]);
  int nl = atoi(argv[5]);
  int nm = atoi(argv[6]);

  double *A = (double *)malloc(ni * nk * sizeof(double));
  double *B = (double *)malloc(nk * nj * sizeof(double));
  double *C = (double *)malloc(nj * nm * sizeof(double));
  double *D = (double *)malloc(nm * nl * sizeof(double));
  double *E = (double *)malloc(ni * nj * sizeof(double));
  double *F = (double *)malloc(nj * nl * sizeof(double));
  double *G = (double *)malloc(ni * nl * sizeof(double));

  init_array(ni, nj, nk, nl, nm, A, B, C, D, E, F, G);

  double *dev_A;
  double *dev_B;
  double *dev_C;
  double *dev_D;
  double *dev_E;
  double *dev_F;
  double *dev_G;
  cudaMalloc(&dev_A, ni * nk * sizeof(double));
  cudaMalloc(&dev_B, nk * nj * sizeof(double));
  cudaMalloc(&dev_C, nl * nj * sizeof(double));
  cudaMalloc(&dev_D, ni * nl * sizeof(double));
  cudaMalloc(&dev_E, ni * nj * sizeof(double));
  cudaMalloc(&dev_F, nj * nl * sizeof(double));
  cudaMalloc(&dev_G, ni * nl * sizeof(double));
  cudaMemcpy(dev_A, A, ni * nk * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, nk * nj * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_C, C, nl * nj * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_D, D, ni * nl * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_E, E, ni * nj * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_F, F, nj * nl * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_G, G, ni * nl * sizeof(double), cudaMemcpyHostToDevice);

  kernel(ni, nj, nk, nl, nm, dev_E, dev_A, dev_B, dev_F, dev_C, dev_D, dev_G);

  cudaMemcpy(G, dev_G, ni * nl * sizeof(double), cudaMemcpyDeviceToHost);

  if (dump_code == 1)
    print_array(ni, nl, G);

  free((void *)E);
  ;
  free((void *)A);
  ;
  free((void *)B);
  ;
  free((void *)F);
  ;
  free((void *)C);
  ;
  free((void *)D);
  ;
  free((void *)G);
  ;

  return 0;
}
