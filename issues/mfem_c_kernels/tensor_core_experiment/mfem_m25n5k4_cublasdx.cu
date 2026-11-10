#include <cuda_runtime.h>
#include <cublasdx.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int M = 25;
constexpr int N = 5;
constexpr int K = 4;
constexpr int NE = 1024;

#define CUDA_OK(expr) do {                                                   \
  cudaError_t error_ = (expr);                                               \
  if (error_ != cudaSuccess) {                                               \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
                 cudaGetErrorString(error_));                                \
    std::exit(1);                                                            \
  }                                                                          \
} while (0)

using BlasDxGemm = decltype(
    cublasdx::Size<M, N, K>() +
    cublasdx::Precision<cublasdx::tfloat32_t,
                        cublasdx::tfloat32_t, float>() +
    cublasdx::Type<cublasdx::type::real>() +
    cublasdx::Function<cublasdx::function::MM>() +
    cublasdx::Arrangement<cublasdx::row_major,
                          cublasdx::row_major,
                          cublasdx::row_major>() +
    cublasdx::Block() + cublasdx::SM<870>());

__global__ void fp32_cuda_core(const float *a, const float *b, float *c) {
  int linear = threadIdx.x;
  if (linear >= M * N) return;
  int row = linear / N;
  int col = linear % N;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < K; ++k)
    sum = fmaf(a[(blockIdx.x * M + row) * K + k], b[k * N + col], sum);
  c[(blockIdx.x * M + row) * N + col] = sum;
}

template<class GEMM>
__launch_bounds__(GEMM::max_threads_per_block)
__global__ void cublasdx_gemm(const typename GEMM::a_value_type *a,
                              const typename GEMM::b_value_type *b,
                              typename GEMM::c_value_type *c) {
  extern __shared__ __align__(16) char smem[];
  a += (size_t)blockIdx.x * M * K;
  c += (size_t)blockIdx.x * M * N;

  auto ag = cublasdx::make_tensor(a, GEMM::get_layout_gmem_a());
  auto bg = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
  auto cg = cublasdx::make_tensor(c, GEMM::get_layout_gmem_c());
  auto [as_ptr, bs_ptr] = cublasdx::slice_shared_memory_ab<GEMM>(smem);
  auto as = cublasdx::make_tensor(as_ptr, GEMM::get_layout_smem_a());
  auto bs = cublasdx::make_tensor(bs_ptr, GEMM::get_layout_smem_b());
  using alignment = cublasdx::alignment_of<GEMM>;
  cublasdx::copy<GEMM, alignment::a>(ag, as);
  cublasdx::copy<GEMM, alignment::b>(bg, bs);
  cublasdx::copy_wait();

  auto accumulator = GEMM::suggest_accumulator();
  GEMM().execute(as, bs, accumulator);
  accumulator.partition_and_store(cg);
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
  using TA = typename BlasDxGemm::a_value_type;
  using TB = typename BlasDxGemm::b_value_type;
  constexpr size_t smem_bytes =
      cublasdx::get_shared_storage_size_ab<BlasDxGemm>();

  std::vector<float> af((size_t)NE * M * K);
  std::vector<float> bf(K * N);
  std::vector<TA> atf32(af.size());
  std::vector<TB> btf32(bf.size());
  for (size_t i = 0; i < af.size(); ++i) {
    af[i] = std::sin((float)(i % 97) * 0.071f) * 0.75f;
    atf32[i] = TA(af[i]);
  }
  for (size_t i = 0; i < bf.size(); ++i) {
    bf[i] = std::cos((float)(i + 3) * 0.113f) * 0.5f;
    btf32[i] = TB(bf[i]);
  }

  float *daf = nullptr, *dbf = nullptr, *dref = nullptr, *dout = nullptr;
  TA *da = nullptr;
  TB *db = nullptr;
  CUDA_OK(cudaMalloc(&daf, af.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&dbf, bf.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&dref, (size_t)NE * M * N * sizeof(float)));
  CUDA_OK(cudaMalloc(&dout, (size_t)NE * M * N * sizeof(float)));
  CUDA_OK(cudaMalloc(&da, atf32.size() * sizeof(TA)));
  CUDA_OK(cudaMalloc(&db, btf32.size() * sizeof(TB)));
  CUDA_OK(cudaMemcpy(daf, af.data(), af.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dbf, bf.data(), bf.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(da, atf32.data(), atf32.size() * sizeof(TA), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(db, btf32.data(), btf32.size() * sizeof(TB), cudaMemcpyHostToDevice));

  std::fprintf(stderr,
               "diagnostic: block=(%u,%u,%u) shared=%zu sizeof(A)=%zu sizeof(B)=%zu\n",
               BlasDxGemm::block_dim.x, BlasDxGemm::block_dim.y,
               BlasDxGemm::block_dim.z, smem_bytes, sizeof(TA), sizeof(TB));

  CUDA_OK(cudaFuncSetAttribute(
      cublasdx_gemm<BlasDxGemm>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  fp32_cuda_core<<<NE, 128>>>(daf, dbf, dref);
  cublasdx_gemm<BlasDxGemm>
      <<<NE, BlasDxGemm::block_dim, smem_bytes>>>(da, db, dout);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());

  std::vector<float> ref((size_t)NE * M * N);
  std::vector<float> out(ref.size());
  CUDA_OK(cudaMemcpy(ref.data(), dref, ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  double max_abs = 0.0, max_rel = 0.0, sum_sq = 0.0, ref_sq = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double error = std::fabs((double)out[i] - ref[i]);
    max_abs = std::fmax(max_abs, error);
    max_rel = std::fmax(max_rel,
                        error / std::fmax(std::fabs((double)ref[i]), 1.0e-12));
    sum_sq += error * error;
    ref_sq += (double)ref[i] * ref[i];
  }
  double rel_l2 = std::sqrt(sum_sq / ref_sq);
  bool pass = max_abs < 5.0e-4 && rel_l2 < 1.0e-3;

  constexpr int iterations = 500;
  float fp32_us = time_kernel([&] {
    fp32_cuda_core<<<NE, 128>>>(daf, dbf, dref);
  }, iterations);
  float dx_us = time_kernel([&] {
    cublasdx_gemm<BlasDxGemm>
        <<<NE, BlasDxGemm::block_dim, smem_bytes>>>(da, db, dout);
  }, iterations);
  CUDA_OK(cudaGetLastError());

  cudaDeviceProp prop{};
  CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device=%s cc=%d.%d\n", prop.name, prop.major, prop.minor);
  std::printf("problem=batch(%d) x (%dx%d * %dx%d)\n", NE, M, K, K, N);
  std::printf("cublasdx=0.5.0 block=(%u,%u,%u) shared_bytes=%zu\n",
              BlasDxGemm::block_dim.x, BlasDxGemm::block_dim.y,
              BlasDxGemm::block_dim.z, smem_bytes);
  std::printf("correctness=max_abs=%.9g max_rel=%.9g rel_l2=%.9g %s\n",
              max_abs, max_rel, rel_l2, pass ? "PASS" : "FAIL");
  std::printf("fp32_cuda_core_us=%.6f\n", fp32_us);
  std::printf("cublasdx_tf32_us=%.6f\n", dx_us);
  std::printf("cublasdx_speedup=%.6fx\n", fp32_us / dx_us);

  cudaFree(daf); cudaFree(dbf); cudaFree(dref); cudaFree(dout);
  cudaFree(da); cudaFree(db);
  return pass ? 0 : 2;
}
