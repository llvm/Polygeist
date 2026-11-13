#include <cuda_runtime.h>
#include <mma.h>
#include <cublasdx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace nvcuda;

#define CUDA_OK(expr) do {                                                   \
  cudaError_t error_ = (expr);                                               \
  if (error_ != cudaSuccess) {                                               \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,               \
                 cudaGetErrorString(error_));                                \
    std::exit(1);                                                            \
  }                                                                          \
} while (0)

template<int M, int N, int K>
__global__ void scalar_gemm(const float* a, const float* b, float* c) {
  for (int linear = threadIdx.x; linear < M * N; linear += blockDim.x) {
    const int row = linear / N;
    const int col = linear % N;
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < K; ++k)
      sum = fmaf(a[((size_t)blockIdx.x * M + row) * K + k],
                 b[k * N + col], sum);
    c[((size_t)blockIdx.x * M + row) * N + col] = sum;
  }
}

__global__ void round_to_tf32(float* data, size_t count) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) data[i] = wmma::__float_to_tf32(data[i]);
}

// One warp computes one padded 16x16 output tile and iterates over K in
// padded k=8 steps through the WMMA API.
template<int M, int N, int K, int MP, int NP, int KP>
__global__ void padded_wmma(const float* a, const float* b, float* c) {
  constexpr int MT = MP / 16;
  constexpr int NT = NP / 16;
  const int matrix = blockIdx.x;
  const int warp = threadIdx.x / 32;
  const int warps = blockDim.x / 32;
  for (int tile = warp; tile < MT * NT; tile += warps) {
    const int tm = tile / NT;
    const int tn = tile % NT;
    wmma::fragment<wmma::matrix_a, 16, 16, 8,
                   wmma::precision::tf32, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 8,
                   wmma::precision::tf32, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    for (int k0 = 0; k0 < KP; k0 += 8) {
      const float* ap = a + ((size_t)matrix * MP + tm * 16) * KP + k0;
      const float* bp = b + k0 * NP + tn * 16;
      wmma::load_matrix_sync(af, ap, KP);
      wmma::load_matrix_sync(bf, bp, NP);
      wmma::mma_sync(cf, af, bf, cf);
    }
    float* cp = c + ((size_t)matrix * MP + tm * 16) * NP + tn * 16;
    wmma::store_matrix_sync(cp, cf, NP, wmma::mem_row_major);
  }
}

// One warp computes one 16x8 tile with Ampere's smallest native TF32 MMA.
// It iterates over K in padded k=4 steps, avoiding WMMA's 16x16x8 tile.
template<int M, int N, int K, int MP, int NP, int KP>
__global__ void direct_mma(const float* a, const float* b, float* c) {
  constexpr int MT = MP / 16;
  constexpr int NT = NP / 8;
  const int matrix = blockIdx.x;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int warps = blockDim.x / 32;
  for (int tile = warp; tile < MT * NT; tile += warps) {
    const int tm = tile / NT;
    const int tn = tile % NT;
    const int group = lane >> 2;
    const int lane4 = lane & 3;
    const int row0 = tm * 16 + group;
    const int row1 = row0 + 8;
    const int col0 = tn * 8 + lane4 * 2;
    const int col1 = col0 + 1;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (int k0 = 0; k0 < KP; k0 += 4) {
      const unsigned a0 = __float_as_uint(
          a[((size_t)matrix * MP + row0) * KP + k0 + lane4]);
      const unsigned a1 = __float_as_uint(
          a[((size_t)matrix * MP + row1) * KP + k0 + lane4]);
      const unsigned b0 = __float_as_uint(
          b[(k0 + lane4) * NP + tn * 8 + group]);
      asm volatile(
          "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
          : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
          : "r"(a0), "r"(a1), "r"(b0));
    }
    const size_t base = (size_t)matrix * M * N;
    if (row0 < M && col0 < N) c[base + row0 * N + col0] = d0;
    if (row0 < M && col1 < N) c[base + row0 * N + col1] = d1;
    if (row1 < M && col0 < N) c[base + row1 * N + col0] = d2;
    if (row1 < M && col1 < N) c[base + row1 * N + col1] = d3;
  }
}

template<class GEMM, int M, int N, int K>
__launch_bounds__(GEMM::max_threads_per_block)
__global__ void dx_gemm(const typename GEMM::a_value_type* a,
                        const typename GEMM::b_value_type* b,
                        typename GEMM::c_value_type* c) {
  extern __shared__ __align__(16) char smem[];
  a += (size_t)blockIdx.x * M * K;
  c += (size_t)blockIdx.x * M * N;
  auto ag = cublasdx::make_tensor(a, GEMM::get_layout_gmem_a());
  auto bg = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
  auto cg = cublasdx::make_tensor(c, GEMM::get_layout_gmem_c());
  auto [asp, bsp] = cublasdx::slice_shared_memory_ab<GEMM>(smem);
  auto as = cublasdx::make_tensor(asp, GEMM::get_layout_smem_a());
  auto bs = cublasdx::make_tensor(bsp, GEMM::get_layout_smem_b());
  using alignment = cublasdx::alignment_of<GEMM>;
  cublasdx::copy<GEMM, alignment::a>(ag, as);
  cublasdx::copy<GEMM, alignment::b>(bg, bs);
  cublasdx::copy_wait();
  auto accumulator = GEMM().execute(as, bs);
  accumulator.partition_and_store(cg);
}

template<typename Launch>
float time_kernel(Launch launch, int iterations) {
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

struct Error {
  double max_abs = 0.0;
  double rel_l2 = 0.0;
};

Error compare_compact(const std::vector<float>& ref,
                      const std::vector<float>& got) {
  double sum_sq = 0.0, ref_sq = 0.0, max_abs = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double e = std::fabs((double)got[i] - ref[i]);
    max_abs = std::max(max_abs, e);
    sum_sq += e * e;
    ref_sq += (double)ref[i] * ref[i];
  }
  return {max_abs, std::sqrt(sum_sq / ref_sq)};
}

template<int M, int N, int MP, int NP>
Error compare_padded(int batch, const std::vector<float>& ref,
                     const std::vector<float>& got) {
  double sum_sq = 0.0, ref_sq = 0.0, max_abs = 0.0;
  for (int e = 0; e < batch; ++e)
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        const float r = ref[((size_t)e * M + m) * N + n];
        const float g = got[((size_t)e * MP + m) * NP + n];
        const double d = std::fabs((double)g - r);
        max_abs = std::max(max_abs, d);
        sum_sq += d * d;
        ref_sq += (double)r * r;
      }
  return {max_abs, std::sqrt(sum_sq / ref_sq)};
}

template<int M, int N, int K, int BATCH, int ITERS>
bool run_shape(const char* kind) {
  constexpr int MP16 = ((M + 15) / 16) * 16;
  constexpr int NP16 = ((N + 15) / 16) * 16;
  constexpr int NP8 = ((N + 7) / 8) * 8;
  constexpr int KP8 = ((K + 7) / 8) * 8;
  constexpr int KP4 = ((K + 3) / 4) * 4;
  using DX = decltype(
      cublasdx::Size<M, N, K>() +
      cublasdx::Precision<cublasdx::tfloat32_t,
                          cublasdx::tfloat32_t, float>() +
      cublasdx::Type<cublasdx::type::real>() +
      cublasdx::Function<cublasdx::function::MM>() +
      cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major,
                            cublasdx::row_major>() +
      cublasdx::Block() + cublasdx::SM<870>());
  using TA = typename DX::a_value_type;
  using TB = typename DX::b_value_type;
  constexpr size_t dx_smem = cublasdx::get_shared_storage_size_ab<DX>();

  std::vector<float> a((size_t)BATCH * M * K), b(K * N);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = std::sin((float)(i % 251) * 0.031f) * 0.75f;
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = std::cos((float)(i % 127 + 3) * 0.047f) * 0.5f;

  std::vector<float> aw((size_t)BATCH * MP16 * KP8, 0.0f);
  std::vector<float> bw(KP8 * NP16, 0.0f);
  std::vector<float> ad((size_t)BATCH * MP16 * KP4, 0.0f);
  std::vector<float> bd(KP4 * NP8, 0.0f);
  for (int e = 0; e < BATCH; ++e)
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < K; ++k) {
        const float x = a[((size_t)e * M + m) * K + k];
        aw[((size_t)e * MP16 + m) * KP8 + k] = x;
        ad[((size_t)e * MP16 + m) * KP4 + k] = x;
      }
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n) {
      bw[k * NP16 + n] = b[k * N + n];
      bd[k * NP8 + n] = b[k * N + n];
    }
  std::vector<TA> ax(a.size());
  std::vector<TB> bx(b.size());
  for (size_t i = 0; i < a.size(); ++i) ax[i] = TA(a[i]);
  for (size_t i = 0; i < b.size(); ++i) bx[i] = TB(b[i]);

  float *da, *db, *dref, *dwmma, *ddirect, *daw, *dbw, *dad, *dbd, *ddx;
  TA* dax; TB* dbx;
  CUDA_OK(cudaMalloc(&da, a.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&db, b.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&dref, (size_t)BATCH * M * N * sizeof(float)));
  CUDA_OK(cudaMalloc(&dwmma, (size_t)BATCH * MP16 * NP16 * sizeof(float)));
  CUDA_OK(cudaMalloc(&ddirect, (size_t)BATCH * M * N * sizeof(float)));
  CUDA_OK(cudaMalloc(&daw, aw.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&dbw, bw.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&dad, ad.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&dbd, bd.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&dax, ax.size() * sizeof(TA)));
  CUDA_OK(cudaMalloc(&dbx, bx.size() * sizeof(TB)));
  CUDA_OK(cudaMalloc(&ddx, (size_t)BATCH * M * N * sizeof(float)));
  CUDA_OK(cudaMemcpy(da, a.data(), a.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(db, b.data(), b.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(daw, aw.data(), aw.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dbw, bw.data(), bw.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dad, ad.data(), ad.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dbd, bd.data(), bd.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dax, ax.data(), ax.size()*sizeof(TA), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dbx, bx.data(), bx.size()*sizeof(TB), cudaMemcpyHostToDevice));
  round_to_tf32<<<(aw.size()+255)/256,256>>>(daw, aw.size());
  round_to_tf32<<<(bw.size()+255)/256,256>>>(dbw, bw.size());
  round_to_tf32<<<(ad.size()+255)/256,256>>>(dad, ad.size());
  round_to_tf32<<<(bd.size()+255)/256,256>>>(dbd, bd.size());
  CUDA_OK(cudaFuncSetAttribute(dx_gemm<DX,M,N,K>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dx_smem));

  constexpr int scalar_threads = std::min(256, ((M*N+31)/32)*32);
  constexpr int wmma_threads = std::min(256, (MP16/16)*(NP16/16)*32);
  constexpr int direct_threads = std::min(256, (MP16/16)*(NP8/8)*32);
  scalar_gemm<M,N,K><<<BATCH,scalar_threads>>>(da,db,dref);
  padded_wmma<M,N,K,MP16,NP16,KP8><<<BATCH,wmma_threads>>>(daw,dbw,dwmma);
  direct_mma<M,N,K,MP16,NP8,KP4><<<BATCH,direct_threads>>>(dad,dbd,ddirect);
  dx_gemm<DX,M,N,K><<<BATCH,DX::block_dim,dx_smem>>>(dax,dbx,ddx);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());

  std::vector<float> ref((size_t)BATCH*M*N), direct(ref.size()), dx(ref.size());
  std::vector<float> wmma((size_t)BATCH*MP16*NP16);
  CUDA_OK(cudaMemcpy(ref.data(),dref,ref.size()*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(wmma.data(),dwmma,wmma.size()*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(direct.data(),ddirect,direct.size()*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(dx.data(),ddx,dx.size()*sizeof(float),cudaMemcpyDeviceToHost));
  const Error ew = compare_padded<M,N,MP16,NP16>(BATCH,ref,wmma);
  const Error ed = compare_compact(ref,direct);
  const Error ex = compare_compact(ref,dx);
  const bool pass = ew.rel_l2 < 0.002 && ed.rel_l2 < 0.002 && ex.rel_l2 < 0.002;

  const float ts = time_kernel([&]{scalar_gemm<M,N,K><<<BATCH,scalar_threads>>>(da,db,dref);},ITERS);
  const float tw = time_kernel([&]{padded_wmma<M,N,K,MP16,NP16,KP8><<<BATCH,wmma_threads>>>(daw,dbw,dwmma);},ITERS);
  const float td = time_kernel([&]{direct_mma<M,N,K,MP16,NP8,KP4><<<BATCH,direct_threads>>>(dad,dbd,ddirect);},ITERS);
  const float tx = time_kernel([&]{dx_gemm<DX,M,N,K><<<BATCH,DX::block_dim,dx_smem>>>(dax,dbx,ddx);},ITERS);
  CUDA_OK(cudaGetLastError());
  std::printf("RESULT kind=%s M=%d N=%d K=%d batch=%d scalar_us=%.6f wmma_us=%.6f direct_us=%.6f cublasdx_us=%.6f wmma_x=%.6f direct_x=%.6f cublasdx_x=%.6f wmma_l2=%.9g direct_l2=%.9g cublasdx_l2=%.9g dx_block=%u dx_smem=%zu status=%s\n",
      kind,M,N,K,BATCH,ts,tw,td,tx,ts/tw,ts/td,ts/tx,ew.rel_l2,ed.rel_l2,ex.rel_l2,
      DX::block_dim.x,dx_smem,pass?"PASS":"FAIL");
  std::fflush(stdout);

  cudaFree(da); cudaFree(db); cudaFree(dref); cudaFree(dwmma); cudaFree(ddirect);
  cudaFree(daw); cudaFree(dbw); cudaFree(dad); cudaFree(dbd);
  cudaFree(dax); cudaFree(dbx); cudaFree(ddx);
  return pass;
}

int main() {
  cudaDeviceProp p{}; CUDA_OK(cudaGetDeviceProperties(&p,0));
  std::printf("device=%s cc=%d.%d cublasdx=0.5.0 cuda_runtime=%d\n",
              p.name,p.major,p.minor,CUDART_VERSION);
  bool ok = true;
  ok &= run_shape<7,5,4,4096,300>("irregular");
  ok &= run_shape<13,9,7,2048,300>("irregular");
  ok &= run_shape<25,5,4,1024,300>("irregular-mfem");
  ok &= run_shape<31,17,9,512,300>("irregular");
  ok &= run_shape<45,27,13,256,250>("irregular");
  ok &= run_shape<8,8,8,2048,300>("regular");
  ok &= run_shape<16,16,8,1024,300>("regular");
  ok &= run_shape<32,32,16,256,250>("regular");
  ok &= run_shape<64,64,32,64,200>("regular");
  ok &= run_shape<128,128,64,16,100>("regular");
  return ok ? 0 : 2;
}
