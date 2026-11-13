// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s
module {
  kernel.defn @cubSegmentedNanSum_f32_memref(
      %input: memref<?x64xf32>, %out: memref<?xf32>) { kernel.yield }
  func.func @nansum_rows(%input: memref<?x64xf32>, %out: memref<?xf32>) {
    kernel.launch @cubSegmentedNanSum_f32_memref(%input, %out)
        {polygeist.fixed_extents = array<i64: 16, 64>} :
        (memref<?x64xf32>, memref<?xf32>) -> ()
    return
  }
}
// CHECK: %[[ROWS:.*]] = arith.constant 16 : i32
// CHECK: %[[COLS:.*]] = arith.constant 64 : i32
// CHECK: %[[OP:.*]] = arith.constant 3 : i32
// CHECK: call @polygeist_cub_segmented_reduce_f32(%[[OP]], %[[ROWS]], %[[COLS]],
// CHECK-NOT: kernel.launch
