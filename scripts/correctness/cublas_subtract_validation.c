#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

void polygeist_cublas_dgemm(int32_t, int32_t, int32_t, double,
                            const double *, int32_t, const double *, int32_t,
                            double, double *, int32_t);
void polygeist_cublas_dgemv_T(int32_t, int32_t, double, const double *,
                              int32_t, const double *, double, double *);
#define CUDA_TRY(call)                                                        \
  do {                                                                        \
    cudaError_t status = (call);                                              \
    if (status != cudaSuccess) {                                              \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(status)); \
      return 1;                                                               \
    }                                                                         \
  } while (0)
#define CUBLAS_TRY(call)                                                   \
  do {                                                                     \
    cublasStatus_t status = (call);                                        \
    if (status != CUBLAS_STATUS_SUCCESS) {                                 \
      fprintf(stderr, "%s failed: cublas status %d\n", #call, (int)status); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

static int close_enough(double actual, double expected) {
  return fabs(actual - expected) <= 1.0e-10;
}

int main(void) {
  const double matrix[6] = {1, 2, 3, 4, 5, 6};
  const double vector[2] = {2, -1};
  double output[3] = {10, 20, 30};
  const double expected_vector[3] = {12, 21, 30};
  double *device_matrix = NULL, *device_vector = NULL, *device_output = NULL;
  CUDA_TRY(cudaMalloc((void **)&device_matrix, sizeof(matrix)));
  CUDA_TRY(cudaMalloc((void **)&device_vector, sizeof(vector)));
  CUDA_TRY(cudaMalloc((void **)&device_output, sizeof(output)));
  CUDA_TRY(cudaMemcpy(device_matrix, matrix, sizeof(matrix),
                      cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(device_vector, vector, sizeof(vector),
                      cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(device_output, output, sizeof(output),
                      cudaMemcpyHostToDevice));
  double copied_matrix[6] = {0}, copied_vector[2] = {0};
  CUDA_TRY(cudaMemcpy(copied_matrix, device_matrix, sizeof(copied_matrix),
                      cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpy(copied_vector, device_vector, sizeof(copied_vector),
                      cudaMemcpyDeviceToHost));
  for (int i = 0; i < 6; ++i)
    if (!close_enough(copied_matrix[i], matrix[i]))
      return 1;
  for (int i = 0; i < 2; ++i)
    if (!close_enough(copied_vector[i], vector[i]))
      return 1;
  // Establish the vendor call's expected row-major-transpose convention on
  // this exact silicon before testing the Polygeist ABI wrapper.
  cublasHandle_t reference_handle = NULL;
  const double minus_one = -1.0, plus_one = 1.0;
  double reference_output[3] = {10, 20, 30};
  CUBLAS_TRY(cublasCreate(&reference_handle));
  CUBLAS_TRY(cublasDgemv(reference_handle, CUBLAS_OP_N, 3, 2, &minus_one,
                         device_matrix, 3, device_vector, 1, &plus_one,
                         device_output, 1));
  CUDA_TRY(cudaDeviceSynchronize());
  CUDA_TRY(cudaMemcpy(reference_output, device_output, sizeof(reference_output),
                      cudaMemcpyDeviceToHost));
  CUBLAS_TRY(cublasDestroy(reference_handle));
  for (int i = 0; i < 3; ++i)
    if (!close_enough(reference_output[i], expected_vector[i])) {
      fprintf(stderr, "raw cublas dgemv mismatch at %d: %.17g != %.17g\n",
              i, reference_output[i], expected_vector[i]);
      return 1;
    }
  CUDA_TRY(cudaMemcpy(device_output, output, sizeof(output),
                      cudaMemcpyHostToDevice));
  polygeist_cublas_dgemv_T(2, 3, -1.0, device_matrix, 3, device_vector, 1.0,
                           device_output);
  CUDA_TRY(cudaDeviceSynchronize());
  CUDA_TRY(cudaMemcpy(output, device_output, sizeof(output),
                      cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaFree(device_matrix));
  CUDA_TRY(cudaFree(device_vector));
  CUDA_TRY(cudaFree(device_output));
  for (int i = 0; i < 3; ++i) {
    if (!close_enough(output[i], expected_vector[i])) {
      fprintf(stderr, "dgemv subtract mismatch at %d: %.17g != %.17g\n",
              i, output[i], expected_vector[i]);
      return 1;
    }
  }

  const double left[4] = {1, 2, 3, 4};
  const double right[4] = {5, 6, 7, 8};
  double destination[4] = {100, 100, 100, 100};
  const double expected_matrix[4] = {81, 78, 57, 50};
  double *device_left = NULL, *device_right = NULL, *device_destination = NULL;
  CUDA_TRY(cudaMalloc((void **)&device_left, sizeof(left)));
  CUDA_TRY(cudaMalloc((void **)&device_right, sizeof(right)));
  CUDA_TRY(cudaMalloc((void **)&device_destination, sizeof(destination)));
  CUDA_TRY(cudaMemcpy(device_left, left, sizeof(left), cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(device_right, right, sizeof(right),
                      cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(device_destination, destination, sizeof(destination),
                      cudaMemcpyHostToDevice));
  polygeist_cublas_dgemm(2, 2, 2, -1.0, device_left, 2, device_right, 2, 1.0,
                         device_destination, 2);
  CUDA_TRY(cudaDeviceSynchronize());
  CUDA_TRY(cudaMemcpy(destination, device_destination, sizeof(destination),
                      cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaFree(device_left));
  CUDA_TRY(cudaFree(device_right));
  CUDA_TRY(cudaFree(device_destination));
  for (int i = 0; i < 4; ++i) {
    if (!close_enough(destination[i], expected_matrix[i])) {
      fprintf(stderr, "dgemm subtract mismatch at %d: %.17g != %.17g\n",
              i, destination[i], expected_matrix[i]);
      return 1;
    }
  }

  puts("PASS: cuBLAS subtract GEMV and GEMM");
  return 0;
}
