// Device image used by mixed_mgpu_graph_smoke.c. This has the same pointer
// ABI as an outlined MLIR gpu.func after GPU-to-NVVM lowering.
extern "C" __global__ void polygeist_generated_scale_kernel(double *values,
                                                             long n,
                                                             double scale) {
  long i = (long)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n)
    values[i] *= scale;
}
