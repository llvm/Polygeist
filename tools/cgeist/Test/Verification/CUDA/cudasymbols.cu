// clang-format off
// RUN: cgeist %s --cuda-lower --cuda-gpu-arch=sm_60 -nocudalib -nocudainc %resourcedir --function=* -S -emit-llvm-dialect -output-intermediate-gpu=1 -emit-cuda -c | FileCheck %s
// TODO only do this test if we have a cuda build

#include "../Inputs/cuda.h"
#include "__clang_cuda_builtin_vars.h"

__device__ float dev_array[2];
__constant__ float const_array[2];

// CHECK: @dev_array = dso_local addrspace(1) externally_initialized global [2 x float] undef
// CHECK: @const_array = dso_local addrspace(4) externally_initialized global [2 x float] undef

// TODO:
// COM: @dev_array = dso_local addrspace(1) externally_initialized global [2 x float] zeroinitializer, align 4
// COM: @const_array = dso_local addrspace(4) externally_initialized global [2 x float] zeroinitializer, align 4

__device__ int dev_i;
__constant__ int const_i;

// CHECK: @dev_i = dso_local addrspace(1) externally_initialized global [1 x i32] undef
// CHECK: @const_i = dso_local addrspace(4) externally_initialized global [1 x i32] undef

// TODO:
// COM: @dev_i = dso_local addrspace(1) externally_initialized global i32 0, align 4
// COM: @const_i = dso_local addrspace(4) externally_initialized global i32 0, align 4


__device__ void use(float a);
__device__ float getf();
__device__ int geti();

int cudaMemcpyToSymbol(void *, void *, size_t);

__global__ void
foo()
{
    use(dev_array[geti()]);
    dev_array[geti()] = getf();

    use(const_array[geti()]);

    use(dev_i);
    dev_i = geti();

    use(const_i);
}

void baz() {
    float af[2] = {1.0f};
    int ai = 3;

    cudaMemcpyToSymbol(dev_array, af, 2 * sizeof(float));
    cudaMemcpyToSymbol(const_array, af, 2 * sizeof(float));

    cudaMemcpyToSymbol(&dev_i, &ai, sizeof(int));
    cudaMemcpyToSymbol(&const_i, &ai, sizeof(int));
}

// When there are multiple callsites that use the same global symbol, the kernel
// outlining will duplicate that into two different gpu.module's, they then need
// to be merged together so that we only get a single global symbol on the gpu
// side too
void bar() {
    foo<<<1,1>>>();
}
void barr() {
    foo<<<1,1>>>();
}

// check that we register _exactly_ 4 symbols

// CHECK-COUNT-4:   llvm.call @__mgpurtRegisterVar(
// CHECK-NOT:   llvm.call @__mgpurtRegisterVar(
