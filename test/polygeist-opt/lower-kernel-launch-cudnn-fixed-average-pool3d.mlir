// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cudnnAveragePool_f32_r5(
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32,
      %input: memref<?x?x?x?x?xf32>,
      %output: memref<?x?x?x?x?xf32>) {
    kernel.yield
  }

  func.func @fixed_average_pool3d(
      %input: memref<?x?x?x?x?xf32>,
      %output: memref<?x?x?x?x?xf32>,
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32) {
    kernel.launch @cudnnAveragePool_f32_r5(
        %operation, %rank, %n, %c, %i0, %i1, %i2, %o0, %o1, %o2,
        %input, %output)
        : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
           memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @fixed_average_pool3d
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: call @polygeist_cudnn_adaptive_pool_f32(
// CHECK-SAME: i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
// CHECK-SAME: !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch
