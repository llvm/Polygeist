#include "polygeist_cublas_rt.h"

#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { MAX_RANK = 64, FIELDS_PER_TENSOR = 3 * MAX_RANK,
       METADATA_SIZE = 3 + 3 * FIELDS_PER_TENSOR };

static void set_tensor_metadata(int64_t metadata[METADATA_SIZE], int tensor,
                                int rank, const int64_t *extents,
                                const int64_t *strides,
                                const int64_t *modes) {
  metadata[tensor] = rank;
  const int base = 3 + tensor * FIELDS_PER_TENSOR;
  for (int dim = 0; dim < rank; ++dim) {
    metadata[base + dim] = extents[dim];
    metadata[base + MAX_RANK + dim] = strides[dim];
    metadata[base + 2 * MAX_RANK + dim] = modes[dim];
  }
}

static int check_cuda(cudaError_t status, const char *operation) {
  if (status == cudaSuccess) return 1;
  fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
  return 0;
}

int main(void) {
  const double host_a[6] = {1, 2, 3, 4, 5, 6};
  const double host_b[6] = {7, 8, 9, 10, 11, 12};
  const double expected[4] = {58, 64, 139, 154};
  double host_c[4] = {0, 0, 0, 0};
  double *device_a = NULL, *device_b = NULL, *device_c = NULL;
  int64_t metadata[METADATA_SIZE];
  memset(metadata, 0, sizeof(metadata));

  const int64_t a_extents[2] = {2, 3};
  const int64_t a_strides[2] = {3, 1};
  const int64_t a_modes[2] = {0, 2};
  const int64_t b_extents[2] = {3, 2};
  const int64_t b_strides[2] = {2, 1};
  const int64_t b_modes[2] = {2, 1};
  const int64_t c_extents[2] = {2, 2};
  const int64_t c_strides[2] = {2, 1};
  const int64_t c_modes[2] = {0, 1};
  set_tensor_metadata(metadata, 0, 2, a_extents, a_strides, a_modes);
  set_tensor_metadata(metadata, 1, 2, b_extents, b_strides, b_modes);
  set_tensor_metadata(metadata, 2, 2, c_extents, c_strides, c_modes);

  if (!check_cuda(cudaMalloc((void **)&device_a, sizeof(host_a)), "cudaMalloc A") ||
      !check_cuda(cudaMalloc((void **)&device_b, sizeof(host_b)), "cudaMalloc B") ||
      !check_cuda(cudaMalloc((void **)&device_c, sizeof(host_c)), "cudaMalloc C") ||
      !check_cuda(cudaMemcpy(device_a, host_a, sizeof(host_a),
                             cudaMemcpyHostToDevice), "copy A") ||
      !check_cuda(cudaMemcpy(device_b, host_b, sizeof(host_b),
                             cudaMemcpyHostToDevice), "copy B"))
    return 2;

  polygeist_cutensornet_contraction2_f64_device(
      device_a, device_b, device_c, metadata);
  polygeist_cutensornet_contraction2_f64_device(
      device_a, device_b, device_c, metadata);
  if (!check_cuda(cudaMemcpy(host_c, device_c, sizeof(host_c),
                             cudaMemcpyDeviceToHost), "copy C"))
    return 2;

  double max_error = 0.0;
  for (int i = 0; i < 4; ++i)
    max_error = fmax(max_error, fabs(host_c[i] - expected[i]));
  printf("cutensornet_device_abi correctness=%s max_error=%.3e "
         "result=[%.1f,%.1f,%.1f,%.1f]\n",
         max_error < 1.0e-12 ? "PASS" : "FAIL", max_error,
         host_c[0], host_c[1], host_c[2], host_c[3]);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  polygeist_cublas_destroy();
  return max_error < 1.0e-12 ? 0 : 1;
}
