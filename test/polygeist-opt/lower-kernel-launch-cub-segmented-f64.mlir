// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s
module {
  kernel.defn @cubSegmentedSum_f64_memref(
      %input: memref<?x?xf64>, %out: memref<?xf64>) { kernel.yield }
  func.func @sum_rows(%input: memref<?x?xf64>, %out: memref<?xf64>) {
    kernel.launch @cubSegmentedSum_f64_memref(%input, %out) :
        (memref<?x?xf64>, memref<?xf64>) -> ()
    return
  }
}
// CHECK: %[[OP:.*]] = arith.constant 0 : i32
// CHECK: call @polygeist_cub_segmented_reduce_f64(%[[OP]],
// CHECK-NOT: kernel.launch
