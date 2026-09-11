// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s
module {
  kernel.defn @cubSegmentedLogicalAnd_i32_memref(
      %x: memref<?x64xi32>, %out: memref<?xi32>) { kernel.yield }
  kernel.defn @cubSegmentedLogicalSelect_i32_memref(
      %x: memref<?x64xi32>, %y: memref<?x64xi32>, %all: i32,
      %out: memref<?xi32>) { kernel.yield }
  func.func @logical(%x: memref<?x64xi32>, %out: memref<?xi32>, %all: i32) {
    kernel.launch @cubSegmentedLogicalAnd_i32_memref(%x, %out) :
        (memref<?x64xi32>, memref<?xi32>) -> ()
    kernel.launch @cubSegmentedLogicalSelect_i32_memref(%x, %x, %all, %out) :
        (memref<?x64xi32>, memref<?x64xi32>, i32, memref<?xi32>) -> ()
    return
  }
}
// CHECK-COUNT-2: call @polygeist_cub_segmented_reduce_i32
// CHECK-NOT: kernel.launch
