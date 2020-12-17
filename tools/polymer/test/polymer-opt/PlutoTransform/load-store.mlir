// RUN: polymer-opt %s -pluto-opt | FileCheck %s

// Test whether a load-store pair can be optimized.

func @load_store() -> () {
  %A = alloc() : memref<64xf32>
  affine.for %i = 0 to 64 {
    %0 = affine.load %A[%i] : memref<64xf32>
    affine.store %0, %A[%i] : memref<64xf32>
  }
  return
}


// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0) -> (d0 * 32 + 32)>
//
//
// CHECK:      func @main(%[[ARG0:.*]]: memref<?xf32>) {
// CHECK-NEXT:   affine.for %[[ARG1:.*]] = 0 to 2 {
// CHECK-NEXT:     affine.for %[[ARG2:.*]] = #[[MAP0]](%[[ARG1]]) to #[[MAP1]](%[[ARG1]]) {
// CHECK-NEXT:       %[[VAL0:.*]] = affine.load %[[ARG0]][%[[ARG2]]] : memref<?xf32>
// CHECK-NEXT:       affine.store %[[VAL0]], %[[ARG0]][%[[ARG2]]] : memref<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
