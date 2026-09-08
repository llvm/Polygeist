#include <cuda_runtime.h>

#include "../../../third_party/gpu-parboil/benchmarks/stencil/src/cuda/kernels.cu"

extern "C" void cpu_stencil(float c0, float c1, float *input, float *output,
                            int nx, int ny, int nz) {
  const int tx = 32, ty = 4;
  dim3 block(tx, ty, 1);
  dim3 grid((nx + tx * 2 - 1) / (tx * 2), (ny + ty - 1) / ty, 1);
  int shared_bytes = tx * 2 * ty * sizeof(float);
  block2D_hybrid_coarsen_x<<<grid, block, shared_bytes>>>(
      c0, c1, input, output, nx, ny, nz);
}
