// RUN: polygeist-opt %s --raise-affine-to-linalg | FileCheck %s

func.func @store_source_index(%input: memref<8xf32>, %output: memref<8xi32>) {
  affine.for %i = 1 to 8 {
    %value = affine.load %input[%i] : memref<8xf32>
    %index = arith.index_cast %i : index to i32
    affine.store %index, %output[%i] : memref<8xi32>
  }
  return
}

// CHECK-LABEL: func.func @store_source_index
// CHECK: %[[RELATIVE:.*]] = linalg.index 0 : index
// CHECK: %[[ABSOLUTE:.*]] = arith.addi %[[RELATIVE]], {{.*}} : index
// CHECK: %[[INDEX:.*]] = arith.index_cast %[[ABSOLUTE]] : index to i32
// CHECK: linalg.yield %[[INDEX]] : i32
