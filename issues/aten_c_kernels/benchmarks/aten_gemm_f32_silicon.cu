// aten_gemm_f32_silicon.cu — f32 (cublasSgemm/Sgemv) silicon measurement for
// the four GEMM/GEMV ATen kernels whose raised path was previously timed in
// FP64 (cublasDgemm/Dgemv). This aligns the "resident CUDA" and "raised"
// columns with the ATen-native column, which torch measured in its default
// FP32. It reuses the exact timing methodology of aten_resident_cuda_baseline.cu
// (warm launch, cudaEvent over BENCH_ITERS, us/iter) so numbers stay comparable.
//
//   resident_us : operands cudaMalloc'd, time only the cuBLAS op (device-resident)
//   raised_us   : operands are host buffers, cudaHostRegister(Mapped) + zero-copy
//                 device pointer (the Jetson host-pointer ABI), time the cuBLAS op
//
// Real cublasSgemm/Sgemv — the same library ATen dispatches. Not a reimplementation.
//
// Build on the Jetson (JetPack nvcc):
//   nvcc -O3 -arch=sm_87 aten_gemm_f32_silicon.cu -lcublas -o aten_gemm_f32_silicon
// Run (4 process repetitions; take warm median of runs 2-4 like the f64 sweep):
//   for r in 1 2 3 4; do ./aten_gemm_f32_silicon; done

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef BENCH_ITERS
#define BENCH_ITERS 20
#endif

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); std::exit(2);} } while(0)
#define BLAS_OK(x) do { cublasStatus_t e=(x); if(e!=CUBLAS_STATUS_SUCCESS){std::fprintf(stderr,"cuBLAS: %d\n",(int)e); std::exit(2);} } while(0)

static double value(size_t i, int salt) {
  int centered = (int)((i * 17 + (size_t)salt * 13 + 5) % 101) - 50;
  return (double)centered / 257.0;
}
static void fill(std::vector<float>& v, int salt) {
  for (size_t i = 0; i < v.size(); ++i) v[i] = (float)value(i, salt);
}

// Same warm-then-event methodology as aten_resident_cuda_baseline.cu::timed.
static float timed(cudaEvent_t begin, cudaEvent_t end,
                   void (*launch)(void*), void* ctx) {
  launch(ctx); CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(begin));
  for (int i = 0; i < BENCH_ITERS; ++i) launch(ctx);
  CUDA_OK(cudaEventRecord(end)); CUDA_OK(cudaEventSynchronize(end));
  float ms = 0; CUDA_OK(cudaEventElapsedTime(&ms, begin, end));
  return 1000.0f * ms / BENCH_ITERS;
}

struct GemmCtx { cublasHandle_t h; const float *a, *b; float *c;
                 int M, N, K; float alpha, beta; bool gemv; };

// Mirrors the f64 baseline's argument order exactly, just Sgemm/Sgemv.
//   gemm: C(MxN) row-major via cublasSgemm(N,N, N,M,K, b,N, a,K, c,N)
//   gemv: y = A^T x + y via cublasSgemv(OP_T, K,M, a,K, x, y)
static void launch_gemm(void* p) {
  GemmCtx* c = (GemmCtx*)p;
  if (c->gemv) {
    BLAS_OK(cublasSgemv(c->h, CUBLAS_OP_T, c->K, c->M, &c->alpha,
                        c->a, c->K, c->b, 1, &c->beta, c->c, 1));
  } else {
    BLAS_OK(cublasSgemm(c->h, CUBLAS_OP_N, CUBLAS_OP_N, c->N, c->M, c->K,
                        &c->alpha, c->b, c->N, c->a, c->K, &c->beta,
                        c->c, c->N));
  }
}

static void run_op(const char* name, int M, int N, int K,
                   float alpha, float beta, bool gemv) {
  // gemv uses A(MxK), x(K), y(M); gemm uses A(MxK), B(KxN), C(MxN).
  size_t na = (size_t)M * K;
  size_t nb = gemv ? (size_t)K : (size_t)K * N;
  size_t nc = gemv ? (size_t)M : (size_t)M * N;
  std::vector<float> ha(na), hb(nb), hc(nc);
  fill(ha, 1); fill(hb, 2); fill(hc, 3);

  cublasHandle_t h; BLAS_OK(cublasCreate(&h));
  cudaEvent_t b0, e0; CUDA_OK(cudaEventCreate(&b0)); CUDA_OK(cudaEventCreate(&e0));

  // ---- resident: cudaMalloc device buffers ----
  float *da, *db, *dc;
  CUDA_OK(cudaMalloc(&da, na * 4)); CUDA_OK(cudaMalloc(&db, nb * 4));
  CUDA_OK(cudaMalloc(&dc, nc * 4));
  CUDA_OK(cudaMemcpy(da, ha.data(), na * 4, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(db, hb.data(), nb * 4, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dc, hc.data(), nc * 4, cudaMemcpyHostToDevice));
  GemmCtx rc{h, da, db, dc, M, N, K, alpha, beta, gemv};
  float resident_us = timed(b0, e0, launch_gemm, &rc);

  // ---- raised host-pointer ABI: cudaHostRegister(Mapped) + zero-copy ----
  std::vector<float> ra(na), rb(nb), rc_h(nc);
  fill(ra, 1); fill(rb, 2); fill(rc_h, 3);
  CUDA_OK(cudaHostRegister(ra.data(), na * 4, cudaHostRegisterMapped));
  CUDA_OK(cudaHostRegister(rb.data(), nb * 4, cudaHostRegisterMapped));
  CUDA_OK(cudaHostRegister(rc_h.data(), nc * 4, cudaHostRegisterMapped));
  float *ma, *mb, *mc;
  CUDA_OK(cudaHostGetDevicePointer(&ma, ra.data(), 0));
  CUDA_OK(cudaHostGetDevicePointer(&mb, rb.data(), 0));
  CUDA_OK(cudaHostGetDevicePointer(&mc, rc_h.data(), 0));
  GemmCtx hc_ctx{h, ma, mb, mc, M, N, K, alpha, beta, gemv};
  float raised_us = timed(b0, e0, launch_gemm, &hc_ctx);

  std::printf("op=%s dtype=f32 M=%d N=%d K=%d resident_us=%.3f raised_us=%.3f\n",
              name, M, N, K, resident_us, raised_us);

  CUDA_OK(cudaHostUnregister(ra.data())); CUDA_OK(cudaHostUnregister(rb.data()));
  CUDA_OK(cudaHostUnregister(rc_h.data()));
  cudaFree(da); cudaFree(db); cudaFree(dc);
  cudaEventDestroy(b0); cudaEventDestroy(e0); cublasDestroy(h);
}

int main() {
  // Shapes/configs taken verbatim from large_problem_comparison.csv f64 rows.
  run_op("aten_mm",    512, 512, 512, 1.0f,  0.0f,  false);  // Dgemm beta=0
  run_op("aten_addmm", 512, 512, 512, 1.25f, 0.5f,  false);  // Dgemm beta=.5 alpha=1.25
  run_op("aten_mv",    4096, 1, 4096, 1.0f,  1.0f,  true);   // Dgemv beta=1 (M4096 K4096)
  run_op("aten_outer", 4096, 4096, 1, 1.0f,  0.0f,  false);  // Dgemm outer product K=1
  return 0;
}
