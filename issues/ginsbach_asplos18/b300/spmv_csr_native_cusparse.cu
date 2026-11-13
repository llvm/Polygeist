#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime_api.h>
#include <cusparse.h>

static void cuda_check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    std::exit(3);
  }
}

static void cusparse_check(cusparseStatus_t status, const char *operation) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
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
  int rows = argc > 1 ? std::atoi(argv[1]) : 1000000;
  int cols = argc > 2 ? std::atoi(argv[2]) : 1000000;
  int entries_per_row = argc > 3 ? std::atoi(argv[3]) : 27;
  int64_t nnz = (int64_t)rows * entries_per_row;
  double *matrix = (double *)std::malloc((size_t)nnz * sizeof(double));
  double *vector = (double *)std::malloc((size_t)cols * sizeof(double));
  double *output = (double *)std::malloc((size_t)rows * sizeof(double));
  int *ranges = (int *)std::malloc(((size_t)rows + 1) * sizeof(int));
  int *indir = (int *)std::malloc((size_t)nnz * sizeof(int));
  for (int i = 0; i <= rows; ++i)
    ranges[i] = i * entries_per_row;
  for (int64_t j = 0; j < nnz; ++j) {
    matrix[j] = 1.0;
    indir[j] = (int)(((uint64_t)j * 17u + (uint64_t)j / entries_per_row) % cols);
  }
  for (int i = 0; i < cols; ++i)
    vector[i] = 1.0;

  double *d_matrix, *d_vector, *d_output;
  int *d_ranges, *d_indir;
  cuda_check(cudaMalloc(&d_matrix, (size_t)nnz * sizeof(double)), "cudaMalloc(matrix)");
  cuda_check(cudaMalloc(&d_vector, (size_t)cols * sizeof(double)), "cudaMalloc(vector)");
  cuda_check(cudaMalloc(&d_output, (size_t)rows * sizeof(double)), "cudaMalloc(output)");
  cuda_check(cudaMalloc(&d_ranges, ((size_t)rows + 1) * sizeof(int)), "cudaMalloc(ranges)");
  cuda_check(cudaMalloc(&d_indir, (size_t)nnz * sizeof(int)), "cudaMalloc(indir)");
  cuda_check(cudaMemcpy(d_matrix, matrix, (size_t)nnz * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy(matrix)");
  cuda_check(cudaMemcpy(d_vector, vector, (size_t)cols * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy(vector)");
  cuda_check(cudaMemcpy(d_ranges, ranges, ((size_t)rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy(ranges)");
  cuda_check(cudaMemcpy(d_indir, indir, (size_t)nnz * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy(indir)");

  cusparseHandle_t handle;
  cusparseSpMatDescr_t mat;
  cusparseDnVecDescr_t x, y;
  cusparse_check(cusparseCreate(&handle), "cusparseCreate");
  cusparse_check(cusparseCreateCsr(&mat, rows, cols, nnz, d_ranges, d_indir,
                                   d_matrix, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                 "cusparseCreateCsr");
  cusparse_check(cusparseCreateDnVec(&x, cols, d_vector, CUDA_R_64F),
                 "cusparseCreateDnVec(x)");
  cusparse_check(cusparseCreateDnVec(&y, rows, d_output, CUDA_R_64F),
                 "cusparseCreateDnVec(y)");
  double alpha = 1.0, beta = 0.0;
  size_t workspace_size = 0;
  cusparse_check(cusparseSpMV_bufferSize(
                     handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, mat, x, &beta, y, CUDA_R_64F,
                     CUSPARSE_SPMV_ALG_DEFAULT, &workspace_size),
                 "cusparseSpMV_bufferSize");
  void *workspace = nullptr;
  cuda_check(cudaMalloc(&workspace, workspace_size), "cudaMalloc(workspace)");
  cudaEvent_t start, stop;
  cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  for (int iteration = 0; iteration < 10; ++iteration) {
    double begin = now_ms();
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    cusparse_check(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, mat, x, &beta, y, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, workspace),
                   "cusparseSpMV");
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

  cuda_check(cudaMemcpy(output, d_output, (size_t)rows * sizeof(double),
                        cudaMemcpyDeviceToHost), "cudaMemcpy(output)");
  double max_abs = 0.0, max_rel = 0.0;
  double expected = (double)entries_per_row;
  for (int i = 0; i < rows; ++i) {
    double abs_error = std::fabs(output[i] - expected);
    double rel_error = abs_error / std::fabs(expected);
    max_abs = abs_error > max_abs ? abs_error : max_abs;
    max_rel = rel_error > max_rel ? rel_error : max_rel;
  }
  bool pass = max_abs <= 1.0e-12 || max_rel <= 1.0e-12;
  std::printf("BENCH_CORRECTNESS max_abs=%.17g max_rel=%.17g atol=1e-12 rtol=1e-12\n",
              max_abs, max_rel);
  std::printf("BENCH_RESULT workload=ginsbach_spmv_csr shape=%dx%d_nnz=%lld status=%s\n",
              rows, cols, (long long)nnz, pass ? "PASS" : "FAIL");

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaFree(workspace);
  cusparseDestroyDnVec(y);
  cusparseDestroyDnVec(x);
  cusparseDestroySpMat(mat);
  cusparseDestroy(handle);
  cudaFree(d_indir);
  cudaFree(d_ranges);
  cudaFree(d_output);
  cudaFree(d_vector);
  cudaFree(d_matrix);
  std::free(indir);
  std::free(ranges);
  std::free(output);
  std::free(vector);
  std::free(matrix);
  return pass ? 0 : 1;
}
