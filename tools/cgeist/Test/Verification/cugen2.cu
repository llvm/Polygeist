// RUN: cgeist %s --cuda-gpu-arch=sm_60 -nocudalib -nocudainc %resourcedir --function=* --cuda-lower --cpuify="distribute" -S | FileCheck %s

#include "Inputs/cuda.h"
#include "__clang_cuda_builtin_vars.h"

#define N 20

__device__ void bar(double* w) {
  w[threadIdx.x] = 2.0;
}

__global__ void foo(double * w) {
  bar(w);
}

void something(double*);

template<typename T>
void templ(T fn, double *w) {
    something(w);
}

void start(double* w) {
  templ(foo, w);
}

// CHECK:   func.func @_Z5startPd(%arg0: memref<?xf64>)
// CHECK-NEXT:     call @_Z9somethingPd(%arg0) : (memref<?xf64>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
