#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace nvcuda;

constexpr int M = 25;
constexpr int N = 5;
constexpr int K = 4;
constexpr int MP = 32;
constexpr int NP = 16;
constexpr int KP = 8;
constexpr int NE = 1024;

#define CUDA_OK(expr) do {                                                   \
  cudaError_t error_ = (expr);                                               \
  if (error_ != cudaSuccess) {                                               \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
                 cudaGetErrorString(error_));                                \
    std::exit(1);                                                            \
  }                                                                          \
} while (0)

__global__ void fp32_cuda_core(const float *a, const float *b, float *c) {
  int element = blockIdx.x;
  int linear = threadIdx.x;
  if (linear >= M * N) return;
  int row = linear / N;
  int col = linear % N;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < K; ++k)
    sum = fmaf(a[(element * M + row) * K + k], b[k * N + col], sum);
  c[(element * M + row) * N + col] = sum;
}

__global__ void round_to_tf32(float *data, size_t count) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) data[i] = wmma::__float_to_tf32(data[i]);
}

__global__ void tf32_tensor_core(const float *a, const float *b, float *c) {
  int warp = threadIdx.x / 32;
  int element = blockIdx.x;
  if (warp >= 2) return;

  wmma::fragment<wmma::matrix_a, 16, 16, 8,
                 wmma::precision::tf32, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 8,
                 wmma::precision::tf32, wmma::row_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> cf;

  const float *ae = a + (size_t)element * MP * KP + warp * 16 * KP;
  float *ce = c + (size_t)element * MP * NP + warp * 16 * NP;
  wmma::load_matrix_sync(af, ae, KP);
  wmma::load_matrix_sync(bf, b, NP);
  wmma::fill_fragment(cf, 0.0f);
  wmma::mma_sync(cf, af, bf, cf);
  wmma::store_matrix_sync(ce, cf, NP, wmma::mem_row_major);
}

// Direct native Ampere TF32 MMA.  This is the Orin analogue of the paper's
// direct m8n8k4 FP64 DMMA approach: use the smallest K=4 instruction, map
// lanes to fragments explicitly, and mask the irregular 25x5 output rather
// than padding all the way to the 16x16x8 WMMA abstraction.
__global__ void tf32_direct_m16n8k4(const float *a, const float *b, float *c) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (warp >= 2) return;

  int group = lane >> 2;
  int thread_in_group = lane & 3;
  int row0 = warp * 16 + group;
  int row1 = row0 + 8;

  // PTX m16n8k4 fragment maps: A owns two rows at one k; B owns one k,n.
  unsigned a0 = __float_as_uint(a[((size_t)blockIdx.x * MP + row0) * KP +
                                  thread_in_group]);
  unsigned a1 = __float_as_uint(a[((size_t)blockIdx.x * MP + row1) * KP +
                                  thread_in_group]);
  unsigned b0 = __float_as_uint(b[thread_in_group * NP + group]);
  float d0, d1, d2, d3;
  asm volatile(
      "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(b0),
        "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));

  int col0 = thread_in_group * 2;
  int col1 = col0 + 1;
  size_t base = (size_t)blockIdx.x * M * N;
  if (row0 < M && col0 < N) c[base + row0 * N + col0] = d0;
  if (row0 < M && col1 < N) c[base + row0 * N + col1] = d1;
  if (row1 < M && col0 < N) c[base + row1 * N + col0] = d2;
  if (row1 < M && col1 < N) c[base + row1 * N + col1] = d3;
}

template <typename Launch>
static float time_kernel(Launch launch, int iterations) {
  cudaEvent_t begin, end;
  CUDA_OK(cudaEventCreate(&begin));
  CUDA_OK(cudaEventCreate(&end));
  for (int i = 0; i < 20; ++i) launch();
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i) launch();
  CUDA_OK(cudaEventRecord(end));
  CUDA_OK(cudaEventSynchronize(end));
  float ms = 0.0f;
  CUDA_OK(cudaEventElapsedTime(&ms, begin, end));
  CUDA_OK(cudaEventDestroy(begin));
  CUDA_OK(cudaEventDestroy(end));
  return 1000.0f * ms / iterations;
}

int main() {
  std::vector<float> a((size_t)NE * M * K);
  std::vector<float> b(K * N);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = std::sin((float)(i % 97) * 0.071f) * 0.75f;
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = std::cos((float)(i + 3) * 0.113f) * 0.5f;

  std::vector<float> ap((size_t)NE * MP * KP, 0.0f);
  std::vector<float> bp(KP * NP, 0.0f);
  for (int e = 0; e < NE; ++e)
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < K; ++k)
        ap[((size_t)e * MP + m) * KP + k] = a[((size_t)e * M + m) * K + k];
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n)
      bp[k * NP + n] = b[k * N + n];

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  float *dap = nullptr, *dbp = nullptr, *dcp = nullptr, *dcdirect = nullptr;
  CUDA_OK(cudaMalloc(&da, a.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&db, b.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&dc, (size_t)NE * M * N * sizeof(float)));
  CUDA_OK(cudaMalloc(&dap, ap.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&dbp, bp.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&dcp, (size_t)NE * MP * NP * sizeof(float)));
  CUDA_OK(cudaMalloc(&dcdirect, (size_t)NE * M * N * sizeof(float)));
  CUDA_OK(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dap, ap.data(), ap.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dbp, bp.data(), bp.size() * sizeof(float), cudaMemcpyHostToDevice));

  round_to_tf32<<<(ap.size() + 255) / 256, 256>>>(dap, ap.size());
  round_to_tf32<<<(bp.size() + 255) / 256, 256>>>(dbp, bp.size());
  CUDA_OK(cudaGetLastError());

  fp32_cuda_core<<<NE, 128>>>(da, db, dc);
  tf32_tensor_core<<<NE, 64>>>(dap, dbp, dcp);
  tf32_direct_m16n8k4<<<NE, 64>>>(dap, dbp, dcdirect);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());

  std::vector<float> cref((size_t)NE * M * N);
  std::vector<float> cpad((size_t)NE * MP * NP);
  std::vector<float> cdirect((size_t)NE * M * N);
  CUDA_OK(cudaMemcpy(cref.data(), dc, cref.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(cpad.data(), dcp, cpad.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(cdirect.data(), dcdirect, cdirect.size() * sizeof(float), cudaMemcpyDeviceToHost));

  double max_abs = 0.0, max_rel = 0.0, sum_sq = 0.0, ref_sq = 0.0;
  double direct_max_abs = 0.0, direct_max_rel = 0.0, direct_sum_sq = 0.0;
  for (int e = 0; e < NE; ++e) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float ref = cref[((size_t)e * M + m) * N + n];
        float got = cpad[((size_t)e * MP + m) * NP + n];
        double error = std::fabs((double)got - ref);
        max_abs = std::fmax(max_abs, error);
        max_rel = std::fmax(max_rel, error / std::fmax(std::fabs((double)ref), 1.0e-12));
        sum_sq += error * error;
        ref_sq += (double)ref * ref;
        float direct = cdirect[((size_t)e * M + m) * N + n];
        double direct_error = std::fabs((double)direct - ref);
        direct_max_abs = std::fmax(direct_max_abs, direct_error);
        direct_max_rel = std::fmax(
            direct_max_rel,
            direct_error / std::fmax(std::fabs((double)ref), 1.0e-12));
        direct_sum_sq += direct_error * direct_error;
      }
    }
  }

  constexpr int iterations = 500;
  float fp32_us = time_kernel([&] {
    fp32_cuda_core<<<NE, 128>>>(da, db, dc);
  }, iterations);
  float tf32_us = time_kernel([&] {
    tf32_tensor_core<<<NE, 64>>>(dap, dbp, dcp);
  }, iterations);
  float direct_us = time_kernel([&] {
    tf32_direct_m16n8k4<<<NE, 64>>>(dap, dbp, dcdirect);
  }, iterations);
  CUDA_OK(cudaGetLastError());

  cudaDeviceProp prop{};
  CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device=%s cc=%d.%d\n", prop.name, prop.major, prop.minor);
  std::printf("problem=batch(%d) x (%dx%d * %dx%d), tensor_tile=16x16x8\n",
              NE, M, K, K, N);
  double rel_l2 = std::sqrt(sum_sq / ref_sq);
  double direct_rel_l2 = std::sqrt(direct_sum_sq / ref_sq);
  bool correctness_pass = max_abs < 5.0e-4 && rel_l2 < 1.0e-3;
  bool direct_correctness_pass =
      direct_max_abs < 5.0e-4 && direct_rel_l2 < 1.0e-3;
  std::printf("correctness=max_abs=%.9g max_rel=%.9g rel_l2=%.9g %s\n",
              max_abs, max_rel, rel_l2,
              correctness_pass ? "PASS" : "FAIL");
  std::printf("direct_correctness=max_abs=%.9g max_rel=%.9g rel_l2=%.9g %s\n",
              direct_max_abs, direct_max_rel, direct_rel_l2,
              direct_correctness_pass ? "PASS" : "FAIL");
  std::printf("fp32_cuda_core_us=%.6f\n", fp32_us);
  std::printf("tf32_tensor_core_us=%.6f\n", tf32_us);
  std::printf("tf32_direct_m16n8k4_us=%.6f\n", direct_us);
  std::printf("wmma_speedup=%.6fx\n", fp32_us / tf32_us);
  std::printf("direct_speedup=%.6fx\n", fp32_us / direct_us);

  cudaFree(da); cudaFree(db); cudaFree(dc);
  cudaFree(dap); cudaFree(dbp); cudaFree(dcp); cudaFree(dcdirect);
  return correctness_pass && direct_correctness_pass ? 0 : 2;
}
