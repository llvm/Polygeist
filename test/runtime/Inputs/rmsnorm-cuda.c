// Hardware regression for the real cuDNN RMSNorm runtime. Cross-compile with
// polygeist_cublas_rt_cuda.c and CUDA/cuDNN; run with POLYGEIST_RT_TIMING=1.
// The runner must reject hostRmsNorm fallback records.
#include "polygeist_cublas_rt.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4096
#define BYTES (N * sizeof(float))
#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); exit(2); \
} } while (0)

static float input[N], weight[N], expected[N], actual[N];

static void values(int iteration) {
  double sum = 0;
  for (int i = 0; i < N; ++i) {
    input[i] = ((i * 7 + iteration * 13) % 97 - 48) / 31.0f;
    weight[i] = ((i * 11 + iteration * 3) % 61 + 1) / 29.0f;
    sum += (double)input[i] * input[i];
  }
  double scale = 1.0 / sqrt(sum / N + 1.e-5);
  for (int i = 0; i < N; ++i)
    expected[i] = input[i] * scale * weight[i];
}

static void check(const char *name) {
  double max_error = 0;
  for (int i = 0; i < N; ++i) {
    double error = fabs(actual[i] - expected[i]);
    if (!isfinite(actual[i]) || error > 2.e-5 + 2.e-5 * fabs(expected[i])) {
      fprintf(stderr, "FAIL %s index=%d actual=%g expected=%g\n",
              name, i, actual[i], expected[i]);
      exit(1);
    }
    if (error > max_error) max_error = error;
  }
  printf("PASS %s max_abs=%.9g\n", name, max_error);
}

int main(void) {
  float *host, *dx, *dw, *dy, *managed;
  if (posix_memalign((void **)&host, 4096, BYTES * 4)) return 2;
  CUDA(cudaMalloc((void **)&dx, BYTES));
  CUDA(cudaMalloc((void **)&dw, BYTES));
  CUDA(cudaMalloc((void **)&dy, BYTES));
  CUDA(cudaMallocManaged((void **)&managed, BYTES * 3, cudaMemAttachGlobal));
  // Offset the operands so their page registrations overlap; register only
  // a small interior section of the weight before passing its full span.
  float *hx = host + 1, *hw = hx + N, *hy = hw + N;
  values(0);
  for (int i = 0; i < N; ++i) { hx[i] = input[i]; hw[i] = weight[i]; }
  polygeist_cublas_sscal(256, 1.0f, hw + 2048);
  // First plan creation happens in a pipeline, exercising cache lifetime.
  polygeist_cublas_pipeline_begin();
  polygeist_rmsnorm_f32(N, hx, hw, hy);
  polygeist_cublas_pipeline_end();
  for (int i = 0; i < N; ++i) actual[i] = hy[i];
  check("partially-registered-host");
  for (int iteration = 1; iteration <= 12; ++iteration) {
    values(iteration);
    for (int i = 0; i < N; ++i) { hx[i] = input[i]; hw[i] = weight[i]; }
    CUDA(cudaMemcpy(dx, input, BYTES, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dw, weight, BYTES, cudaMemcpyHostToDevice));
    polygeist_cublas_pipeline_begin();
    // Alias saxpby requires temporary storage, exercising pool reuse.
    polygeist_cublas_saxpby(N, 0.5f, dx, 0.5f, dx);
    polygeist_rmsnorm_f32(N, dx, dw, dy);
    polygeist_cublas_pipeline_end();
    CUDA(cudaMemcpy(actual, dy, BYTES, cudaMemcpyDeviceToHost));
    check("device-repeated-pipeline");
    polygeist_cublas_pipeline_begin();
    // One-node cuDNN ADD graph, reusing a cached plan across invocations.
    polygeist_cudnn_pointwise_graph_f32(
        N, 16777472, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        hx, hw, NULL, NULL, dy);
    polygeist_cublas_pipeline_end();
    CUDA(cudaMemcpy(actual, dy, BYTES, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) expected[i] = input[i] + weight[i];
    check("cached-pointwise-pipeline");
    values(iteration);
    polygeist_rmsnorm_f32(N, hx, dw, hx);
    for (int i = 0; i < N; ++i) actual[i] = hx[i];
    check("mixed-alias-input");
    polygeist_rmsnorm_f32(N, dx, hw, hw);
    for (int i = 0; i < N; ++i) actual[i] = hw[i];
    check("mixed-alias-weight");
  }
  values(23);
  for (int i = 0; i < N; ++i) { managed[i] = input[i]; managed[N+i] = weight[i]; }
  polygeist_rmsnorm_f32(N, managed, managed + N, managed + 2*N);
  for (int i = 0; i < N; ++i) actual[i] = managed[2*N+i];
  check("managed");
  CUDA(cudaFree(dx)); CUDA(cudaFree(dw)); CUDA(cudaFree(dy));
  CUDA(cudaFree(managed));
  // Runtime retains host registrations for the process lifetime.
  puts("PASS all RMSNorm CUDA regression cases");
  return 0;
}
