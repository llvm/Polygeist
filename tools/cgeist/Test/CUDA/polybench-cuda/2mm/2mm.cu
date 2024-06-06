// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 10 10 10 10
// clang-format on
/**
 * 2mm.c: This file is part of the PolyBench/C 3.2 test suite.
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

__global__ void kernel_A_mul_B(int ni, int nj, int nk, int nl, double alpha,
                               double beta, double *tmp, double *A, double *B,
                               double *C, double *D) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k;
  double dot = 0.0;

  if (i < ni && j < nj) {
    for (k = 0; k < nk; k++)
      dot += alpha * A[i * nk + k] * B[k * nj + j];
    tmp[i * nj + j] = dot;
  }
}

__global__ void kernel_D_plus_tmp_mul_C(int ni, int nj, int nk, int nl,
                                        double alpha, double beta, double *tmp,
                                        double *A, double *B, double *C,
                                        double *D) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l = blockDim.y * blockIdx.y + threadIdx.y;
  int j;
  double dot = 0.0;

  if (i < ni && l < nl) {
    // D[i * nj + l] *= beta;
    dot = D[i * nj + l] * beta;

    for (j = 0; j < nj; j++)
      // D[i * nl + l] += tmp[i * nj + j] * C[j * nl + l];
      dot += tmp[i * nj + j] * C[j * nl + l];
    D[i * nl + l] = dot;
  }
}

short num_blocks(short num, short factor) {
  return (num + factor - 1) / factor;
}

static void kernel(int ni, int nj, int nk, int nl, double alpha, double beta,
                   double *tmp, double *A, double *B, double *C, double *D) {

  unsigned threadsPerBlock = 256;
  dim3 block(threadsPerBlock / 32, 32, 1);

  {
    dim3 grid(num_blocks(ni, block.x), num_blocks(nj, block.y), 1);
    kernel_A_mul_B<<<grid, block>>>(ni, nj, nk, nl, alpha, beta, tmp, A, B, C,
                                    D);
  }

  {
    dim3 grid(num_blocks(ni, block.x), num_blocks(nl, block.y), 1);
    kernel_D_plus_tmp_mul_C<<<grid, block>>>(ni, nj, nk, nl, alpha, beta, tmp,
                                             A, B, C, D);
  }
}

static void print_array(int ni, int nl, double *D) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      fprintf(stderr, "%0.2lf ", D[i * ni + j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

static void init_array(int ni, int nj, int nk, int nl, double *A, double *B,
                       double *C, double *D, double *tmp) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i * ni + j] = ((double)i * j) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i * nk + j] = ((double)i * (j + 1)) / nj;
  for (i = 0; i < nl; i++)
    for (j = 0; j < nj; j++)
      C[i * nl + j] = ((double)i * (j + 3)) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i * ni + j] = ((double)i * (j + 2)) / nk;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      tmp[i * ni + j] = 0;
}

int main(int argc, char **argv) {
  int dump_code = atoi(argv[1]);
  long ni = atoi(argv[2]);
  long nj = atoi(argv[3]);
  long nk = atoi(argv[4]);
  long nl = atoi(argv[5]);

  double alpha = 32412;
  double beta = 2123;
  double *A = (double *)malloc(ni * nk * sizeof(double));
  double *B = (double *)malloc(nk * nj * sizeof(double));
  double *C = (double *)malloc(nl * nj * sizeof(double));
  double *D = (double *)malloc(ni * nl * sizeof(double));
  double *tmp = (double *)malloc(ni * nj * sizeof(double));

  init_array(ni, nj, nk, nl, A, B, C, D, tmp);

  double *dev_A;
  double *dev_B;
  double *dev_C;
  double *dev_D;
  double *dev_tmp;
  double *dev_alpha;
  double *dev_beta;
  cudaMalloc(&dev_A, ni * nk * sizeof(double));
  cudaMalloc(&dev_B, nk * nj * sizeof(double));
  cudaMalloc(&dev_C, nl * nj * sizeof(double));
  cudaMalloc(&dev_D, ni * nl * sizeof(double));
  cudaMalloc(&dev_tmp, ni * nj * sizeof(double));
  cudaMemcpy(dev_A, A, ni * nk * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, nk * nj * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_C, C, nl * nj * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_D, D, ni * nl * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_tmp, tmp, ni * nj * sizeof(double), cudaMemcpyHostToDevice);

  kernel(ni, nj, nk, nl, alpha, beta, dev_tmp, dev_A, dev_B, dev_C, dev_D);

  cudaMemcpy(D, dev_D, ni * nl * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree((void *)dev_A);
  cudaFree((void *)dev_B);
  cudaFree((void *)dev_C);
  cudaFree((void *)dev_D);
  cudaFree((void *)dev_tmp);
  cudaFree((void *)dev_alpha);
  cudaFree((void *)dev_beta);

  if (dump_code == 1)
    print_array(ni, nk, D);

  free((void *)tmp);
  ;
  free((void *)A);
  ;
  free((void *)B);
  ;
  free((void *)C);
  ;
  free((void *)D);
  ;

  return 0;
}
