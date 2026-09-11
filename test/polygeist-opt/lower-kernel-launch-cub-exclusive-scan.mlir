// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cubExclusiveSum1D_i32_memref(
      %input: memref<?xi32>, %output: memref<?xi32>) { kernel.yield }
  func.func @scan(%input: memref<?xi32>, %output: memref<?xi32>) {
    kernel.launch @cubExclusiveSum1D_i32_memref(%input, %output)
        : (memref<?xi32>, memref<?xi32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @scan
// CHECK: call @polygeist_cub_exclusive_sum1d_i32
// CHECK-SAME: (i32, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch
