// clang-format off
// XFAIL: *
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 10
// clang-format on
// This program performs sum reduction with an optimization
// removing warp divergence
// By: Nick from CoffeeBeforeArch

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SHMEM_SIZE 256

__global__ void sumReduction(int *v, int *v_r) {
  // Allocate shared memory
  __shared__ int partial_sum[SHMEM_SIZE];

  // Calculate thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Load elements into shared memory
  partial_sum[threadIdx.x] = v[tid];
  __syncthreads();

  // Increase the stride of the access until we exceed the CTA dimensions
  for (int s = 1; s < blockDim.x; s *= 2) {
    // Change the indexing to be sequential threads
    int index = 2 * s * threadIdx.x;

    // Each thread does work unless the index goes off the block
    if (index < blockDim.x) {
      partial_sum[index] += partial_sum[index + s];
    }
    __syncthreads();
  }

  // Let the thread 0 for this block write it's result to main memory
  // Result is inexed by this block
  if (threadIdx.x == 0) {
    v_r[blockIdx.x] = partial_sum[0];
  }
}

void initialize_vector(int *v, int n) {
  for (int i = 0; i < n; i++) {
    v[i] = 1; // rand() % 10;
  }
}

int main() {
  // Vector size
  int N = 1 << 16;
  int bytes = N * sizeof(int);

  // Original vector and result vector
  int *h_v, *h_v_r;
  int *d_v, *d_v_r;

  // Initialize vector
  initialize_vector(h_v, N);

  // Allocate device memory
  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_v_r, bytes);

  // Copy to device
  cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

  // TB Size
  const int TB_SIZE = 256;

  // Grid Size (No padding)
  int GRID_SIZE = N / TB_SIZE;

  // Call kernels
  sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);

  sumReduction<<<1, TB_SIZE>>>(d_v_r, d_v_r);

  // Copy to host;
  cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

  printf("COMPLETED SUCCESSFULLY %d\n", h_v_r[0]);

  return 0;
}
