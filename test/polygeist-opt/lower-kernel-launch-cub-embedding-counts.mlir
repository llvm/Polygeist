// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cubHistogramEvenI32ShiftZero_memref(
      %samples: memref<?xi32>, %histogram: memref<?xi32>, %shift: i32) {
    kernel.yield
  }
  func.func @embedding_counts(%samples: memref<?xi32>,
                              %histogram: memref<?xi32>, %shift: i32) {
    kernel.launch @cubHistogramEvenI32ShiftZero_memref(
        %samples, %histogram, %shift)
        {polygeist.fixed_extents = array<i64: 512>} :
        (memref<?xi32>, memref<?xi32>, i32) -> ()
    return
  }
}

// CHECK-LABEL: func.func @embedding_counts
// CHECK: %[[COUNT:.+]] = arith.constant 512 : i32
// CHECK: call @polygeist_cub_histogram_even_i32_shift_zero(%[[COUNT]],
// CHECK-NOT: kernel.launch
