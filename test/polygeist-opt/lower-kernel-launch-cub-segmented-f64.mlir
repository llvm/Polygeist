// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s
module {
  kernel.defn @cubSegmentedSum_f64_memref(
      %input: memref<?x?xf64>, %out: memref<?xf64>) { kernel.yield }
  func.func @sum_rows(%input: memref<?x?xf64>, %out: memref<?xf64>) {
    kernel.launch @cubSegmentedSum_f64_memref(%input, %out)
        {polygeist.fixed_extents = array<i64: 16, 64>} :
        (memref<?x?xf64>, memref<?xf64>) -> ()
    return
  }
}
// CHECK: %[[ROWS:.*]] = arith.constant 16 : i32
// CHECK: %[[COLS:.*]] = arith.constant 64 : i32
// CHECK: %[[OP:.*]] = arith.constant 0 : i32
// CHECK: call @polygeist_cub_segmented_reduce_f64(%[[OP]], %[[ROWS]], %[[COLS]],
// CHECK-NOT: kernel.launch
