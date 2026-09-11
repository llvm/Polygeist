// Thin adapter from Polygeist's packed 2D stencil ABI to the external cuSten
// library. This file performs allocation/copy/descriptor plumbing only; the
// stencil computation is implemented by upstream cuSten.

#include "cuSten.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK_CUSTEN(call) do {                                         \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status));                              \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

static int block_divisor(int extent, int preferred) {
  int block = extent < preferred ? extent : preferred;
  while (block > 1 && extent % block != 0)
    --block;
  return block;
}

// cuSten predates Tegra's documented restriction that managed-memory
// prefetch is unsupported when concurrentManagedAccess is false. Its kernels
// work with managed allocations on these systems, but the unconditional
// prefetch leaves CUDA_ERROR_INVALID_DEVICE pending. The cross-build renames
// cuSten's prefetch calls to this compatibility wrapper.
#ifdef cudaMemPrefetchAsync
#undef cudaMemPrefetchAsync
#endif
extern "C" cudaError_t cudaMemPrefetchAsync(
    const void *devPtr, size_t count, int dstDevice, cudaStream_t stream);
extern "C" cudaError_t polygeist_custen_prefetch(
    const void *ptr, size_t bytes, int destination, cudaStream_t stream) {
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess)
    return status;
  int concurrent = 0;
  status = cudaDeviceGetAttribute(
      &concurrent, cudaDevAttrConcurrentManagedAccess, device);
  if (status != cudaSuccess)
    return status;
  if (!concurrent)
    return cudaSuccess;
  return cudaMemPrefetchAsync(ptr, bytes, destination, stream);
}

extern "C" void polygeist_custen_stencil2d_xy_f64(
    int32_t M, int32_t N, int32_t K,
    const double *W, const double *A, double *B) {
  if (M <= 0 || N <= 0 || K <= 0 || (K & 1) == 0 || K > M || K > N ||
      !W || !A || !B) {
    std::fprintf(stderr, "Polygeist cuSten: invalid 2D stencil operands\n");
    std::abort();
  }

  const size_t grid_bytes =
      static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(double);
  const size_t weight_bytes =
      static_cast<size_t>(K) * static_cast<size_t>(K) * sizeof(double);
  double *managed_input = nullptr;
  double *managed_output = nullptr;
  double *managed_weights = nullptr;
  CUDA_CHECK_CUSTEN(cudaMallocManaged(&managed_input, grid_bytes));
  CUDA_CHECK_CUSTEN(cudaMallocManaged(&managed_output, grid_bytes));
  CUDA_CHECK_CUSTEN(cudaMallocManaged(&managed_weights, weight_bytes));
  CUDA_CHECK_CUSTEN(
      cudaMemcpy(managed_input, A, grid_bytes, cudaMemcpyDefault));
  CUDA_CHECK_CUSTEN(cudaMemset(managed_output, 0, grid_bytes));
  CUDA_CHECK_CUSTEN(
      cudaMemcpy(managed_weights, W, weight_bytes, cudaMemcpyDefault));

  int device = 0;
  CUDA_CHECK_CUSTEN(cudaGetDevice(&device));
  const int radius = K / 2;
  // Upstream's 2D-XY implementation and examples are tuned/tested with 4x4
  // blocks; larger blocks expose boundary-indexing bugs in the 2019 kernel.
  const int block_x = block_divisor(N, 4);
  const int block_y = block_divisor(M, 4);
  const int tiles = 1;
  cuSten_t stencil;
  cuStenCreate2DXYnp(
      &stencil, device, tiles, /*nx=*/N, /*ny=*/M,
      block_x, block_y, managed_output, managed_input, managed_weights,
      K, radius, radius, K, radius, radius);
  CUDA_CHECK_CUSTEN(cudaDeviceSynchronize());
  cuStenCompute2DXYnp(&stencil, DEVICE);
  CUDA_CHECK_CUSTEN(cudaDeviceSynchronize());
  // MLIR's memref base-pointer extraction preserves the allocation base (the
  // subview offset is represented separately). cuSten also stores valid
  // values in the interior of its full MxN output, so copy interior-to-
  // interior and preserve the caller's boundary cells.
  const int valid_m = M - 2 * radius;
  const int valid_n = N - 2 * radius;
  for (int row = 0; row < valid_m; ++row) {
    CUDA_CHECK_CUSTEN(cudaMemcpy(
        B + static_cast<size_t>(row + radius) * N + radius,
        managed_output + static_cast<size_t>(row + radius) * N + radius,
        static_cast<size_t>(valid_n) * sizeof(double), cudaMemcpyDefault));
  }
  cuStenDestroy2DXYnp(&stencil);
  CUDA_CHECK_CUSTEN(cudaFree(managed_weights));
  CUDA_CHECK_CUSTEN(cudaFree(managed_output));
  CUDA_CHECK_CUSTEN(cudaFree(managed_input));
}
