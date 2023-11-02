// RUN: clang++ -c %s -o %s.device.ll %stdinclude --cuda-gpu-arch=sm_80 -emit-llvm --cuda-device-only -Xclang -load -Xclang ../Polygeist/build.debug/lib/PolygeistCUDALaunchFixUp.so -Xclang -fpass-plugin=../Polygeist/build.debug/lib/PolygeistCUDALaunchFixUp.so
// RUN: clang++ -c %s -o %s.host.ll %stdinclude --cuda-gpu-arch=sm_80 -emit-llvm --cuda-host-only -Xclang -load -Xclang ../Polygeist/build.debug/lib/PolygeistCUDALaunchFixUp.so -Xclang -fpass-plugin=../Polygeist/build.debug/lib/PolygeistCUDALaunchFixUp.so
// RUN: mlir-translate --import-llvm %s.device.ll -o %s.device.mlir
// RUN: mlir-translate --import-llvm %s.host.ll -o %s.host.mlir
// RUN: polygeist-merge-gpu-modules --host %s.host.mlir --device %s.device.mlir -o %s.merged.mlir
// RUN: car %s.merged.mlir | FileCheck %s
//
// To build an executable:
// mlir-opt %s.merged.mlir --pass-pipeline="builtin.module(gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary{format=binary} )" -o %s.merged-nvvm.mlir
// mlir-translate merged-nvvm.mlir --mlir-to-llvmir -o %smerged.ll
// clang++ %s.merged.ll -O2 -o %s.merged -lcudart_static libmlir_cuda_runtime.so

#include <stdio.h>
#include <cuda_runtime.h>

extern "C" __global__ void foo(float *A)
{
  A[blockIdx.x * blockDim.x + threadIdx.x] = 10;
}

int main(void)
{
  float *C = NULL;
  cudaMalloc((void **)&C, 1024 * 10 * sizeof(float));
  foo<<<1, 1>>>(C);
  cudaMemcpy(&D, C, sizeof(float), cudaMemcpyDeviceToHost);
  printf("res %f\n", D);
  return 0;
}
