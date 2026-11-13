// Fixed C ABI for the actual public Thrust transform/divides implementation.
// Inputs/output are contiguous device-visible FP32 arrays; caller marshals views.
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "polygeist_cuda_graph_rt.h"

template<class Left, class Right>
static void thrust_divide(int32_t n, Left left, Right right, float *output) {
  if (n <= 0 || !output) std::abort();
  const char *trace = std::getenv("POLYGEIST_V2_TRACE");
  if (trace && trace[0] != '0')
    std::fprintf(stderr, "POLYGEIST_VENDOR_CALL library=Thrust entry=transform functor=divides_f32 n=%d\n", n);
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::transform(thrust::cuda::par_nosync.on(stream), left, left+n, right,
                    output, cuda::std::divides<float>{});
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist Thrust transform: %s\n", cudaGetErrorString(status));
    std::abort();
  }
}
extern "C" void polygeist_thrust_divide_f32(int32_t n,const float*x,const float*y,float*out) {
  thrust_divide(n,x,y,out);
}
extern "C" void polygeist_thrust_divide_scalar_left_f32(int32_t n,float x,const float*y,float*out) {
  thrust_divide(n,thrust::make_constant_iterator(x),y,out);
}
extern "C" void polygeist_thrust_divide_scalar_right_f32(int32_t n,const float*x,float y,float*out) {
  thrust_divide(n,x,thrust::make_constant_iterator(y),out);
}
extern "C" void polygeist_thrust_divide_scalars_f32(int32_t n,float x,float y,float*out) {
  thrust_divide(n,thrust::make_constant_iterator(x),thrust::make_constant_iterator(y),out);
}
