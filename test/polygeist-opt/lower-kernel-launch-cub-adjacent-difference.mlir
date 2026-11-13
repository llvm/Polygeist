// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cubAdjacentDifference_f32_memref(
      %input: memref<?xf32>, %out: memref<?xf32>) { kernel.yield }
  func.func @adjacent_difference(%input: memref<?xf32>,
                                 %out: memref<?xf32>) {
    kernel.launch @cubAdjacentDifference_f32_memref(%input, %out)
        {polygeist.fixed_extents = array<i64: 128>} :
        (memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @adjacent_difference
// CHECK: %[[COUNT:.+]] = arith.constant 128 : i32
// CHECK: call @polygeist_cub_adjacent_difference_f32(%[[COUNT]],
// CHECK-NOT: kernel.launch
