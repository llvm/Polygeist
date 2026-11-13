#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

static void cuda_check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    std::exit(3);
  }
}

static void cublas_check(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s failed: %d\n", operation, (int)status);
    std::exit(3);
  }
}

static double now_ms() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (double)ts.tv_sec * 1.0e3 + (double)ts.tv_nsec * 1.0e-6;
}

int main(int argc, char **argv) {
  int n = argc > 1 ? std::atoi(argv[1]) : 512;
  int m = argc > 2 ? std::atoi(argv[2]) : 512;
  int k = argc > 3 ? std::atoi(argv[3]) : 512;
  size_t left_count = (size_t)n * k;
  size_t right_count = (size_t)k * m;
  size_t output_count = (size_t)n * m;
  double *left = (double *)std::malloc(left_count * sizeof(double));
  double *right = (double *)std::malloc(right_count * sizeof(double));
  double *output = (double *)std::malloc(output_count * sizeof(double));
  for (size_t i = 0; i < left_count; ++i)
    left[i] = 1.0;
  for (size_t i = 0; i < right_count; ++i)
    right[i] = 1.0;

  double *d_left, *d_right, *d_output;
  cuda_check(cudaMalloc(&d_left, left_count * sizeof(double)), "cudaMalloc(left)");
  cuda_check(cudaMalloc(&d_right, right_count * sizeof(double)), "cudaMalloc(right)");
  cuda_check(cudaMalloc(&d_output, output_count * sizeof(double)), "cudaMalloc(output)");
  cuda_check(cudaMemcpy(d_left, left, left_count * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy(left)");
  cuda_check(cudaMemcpy(d_right, right, right_count * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy(right)");
  cublasHandle_t handle;
  cublas_check(cublasCreate(&handle), "cublasCreate");
  cudaEvent_t start, stop;
  cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  double alpha = 1.0, beta = 0.0;

  for (int iteration = 0; iteration < 10; ++iteration) {
    double begin = now_ms();
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    cublas_check(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k, &alpha, d_right, m, d_left, k,
                             &beta, d_output, m), "cublasDgemm");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float device_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&device_ms, start, stop),
               "cudaEventElapsedTime");
    double wall_ms = now_ms() - begin;
    std::printf("BENCH_SAMPLE phase=%s index=%d wall_ms=%.9f device_ms=%.9f\n",
                iteration < 5 ? "warmup" : "sample",
                iteration < 5 ? iteration : iteration - 5,
                wall_ms, (double)device_ms);
  }

  cuda_check(cudaMemcpy(output, d_output, output_count * sizeof(double),
                        cudaMemcpyDeviceToHost), "cudaMemcpy(output)");
  double max_abs = 0.0, max_rel = 0.0, expected = (double)k;
  for (size_t i = 0; i < output_count; ++i) {
    double abs_error = std::fabs(output[i] - expected);
    double rel_error = abs_error / std::fabs(expected);
    max_abs = abs_error > max_abs ? abs_error : max_abs;
    max_rel = rel_error > max_rel ? rel_error : max_rel;
  }
  bool pass = max_abs <= 1.0e-9 || max_rel <= 1.0e-9;
  std::printf("BENCH_CORRECTNESS max_abs=%.17g max_rel=%.17g atol=1e-9 rtol=1e-9\n",
              max_abs, max_rel);
  std::printf("BENCH_RESULT workload=ginsbach_gemm shape=%dx%dx%d status=%s\n",
              n, m, k, pass ? "PASS" : "FAIL");

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cublasDestroy(handle);
  cudaFree(d_output);
  cudaFree(d_right);
  cudaFree(d_left);
  std::free(output);
  std::free(right);
  std::free(left);
  return pass ? 0 : 1;
}
