// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s
module {
  kernel.defn @cubSegmentedArgMax_f32_i32_memref(
      %x: memref<?x64xf32>, %out: memref<?xi32>) { kernel.yield }
  kernel.defn @cubSegmentedArgMin_f32_i32_memref(
      %x: memref<?x64xf32>, %out: memref<?xi32>) { kernel.yield }
  func.func @argreduce(%x: memref<?x64xf32>, %out: memref<?xi32>) {
    kernel.launch @cubSegmentedArgMax_f32_i32_memref(%x, %out) :
        (memref<?x64xf32>, memref<?xi32>) -> ()
    kernel.launch @cubSegmentedArgMin_f32_i32_memref(%x, %out) :
        (memref<?x64xf32>, memref<?xi32>) -> ()
    return
  }
}
// CHECK-COUNT-2: call @polygeist_cub_segmented_argreduce_f32
// CHECK-NOT: kernel.launch
