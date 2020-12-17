// RUN: polymer-opt %s -pluto-opt | FileCheck %s

// Test tiling nested loops.

func @load_store_nested_tiling() -> () {
  %A = alloc() : memref<64x64x64xf32>

  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %0 = affine.load %A[%i, %j, %k] : memref<64x64x64xf32>
        affine.store %0, %A[%i, %j, %k] : memref<64x64x64xf32>
      }
    }
  }
  return
}

// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0) -> (d0 * 32 + 31)>
//
//
// CHECK:      func @main(%[[ARG0:.*]]: memref<?x?x?xf32>) {
// CHECK-NEXT:   affine.for %[[ARG1:.*]] = 0 to 1 {
// CHECK-NEXT:     affine.for %[[ARG2:.*]] = 0 to 1 {
// CHECK-NEXT:       affine.for %[[ARG3:.*]] = 0 to 1 {
// CHECK-NEXT:         affine.for %[[ARG4:.*]] = #[[MAP1]](%[[ARG1]]) to #[[MAP2]](%[[ARG1]]) {
// CHECK-NEXT:           affine.for %[[ARG5:.*]] = #[[MAP1]](%[[ARG2]]) to #[[MAP2]](%[[ARG2]]) {
// CHECK-NEXT:             affine.for %[[ARG6:.*]] = #[[MAP1]](%[[ARG3]]) to #[[MAP2]](%[[ARG3]]) {
// CHECK-NEXT:               %[[VAL0:.*]] = affine.load %[[ARG0]][%[[ARG4]], %[[ARG5]], %[[ARG6]]] : memref<?x?x?xf32>
// CHECK-NEXT:               affine.store %[[VAL0]], %[[ARG0]][%[[ARG4]], %[[ARG5]], %[[ARG6]]] : memref<?x?x?xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
