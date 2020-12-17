// RUN: polymer-opt %s -pluto-opt | FileCheck %s

// This case shows how a single loop with complicated arithmetic body can be optimized (blocked).

func @arith() {
  %A = alloc() : memref<64xf32>
  %B = alloc() : memref<64xf32>
  %C = alloc() : memref<64xf32>
  %D = alloc() : memref<64xf32>

  affine.for %i = 0 to 64 {
    %0 = affine.load %A[%i] : memref<64xf32>
    %1 = affine.load %B[%i] : memref<64xf32>
    %2 = addf %0, %1 : f32
    %3 = mulf %0, %2 : f32
    affine.store %2, %C[%i] : memref<64xf32>
    affine.store %3, %D[%i] : memref<64xf32>
  }
  
  return
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0) -> (d0 * 32)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0) -> (d0 * 32 + 31)>
//
//
// CHECK: module {
// CHECK:   func @main(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: memref<?xf32>) {
// CHECK:     affine.for %[[ARG4:.*]] = 0 to 1 {
// CHECK:       affine.for %[[ARG5:.*]] = #[[MAP0]](%[[ARG4]]) to #[[MAP1]](%[[ARG4]]) {
// CHECK:         %[[VAL0:.*]] = affine.load %[[ARG2]][%[[ARG5]]] : memref<?xf32>
// CHECK:         %[[VAL1:.*]] = affine.load %[[ARG1]][%[[ARG5]]] : memref<?xf32>
// CHECK:         %[[VAL2:.*]] = addf %[[VAL1]], %[[VAL0]] : f32
// CHECK:         affine.store %[[VAL2]], %[[ARG0]][%[[ARG5]]] : memref<?xf32>
// CHECK:         %[[VAL3:.*]] = affine.load %[[ARG2]][%[[ARG5]]] : memref<?xf32>
// CHECK:         %[[VAL4:.*]] = addf %[[VAL1]], %[[VAL3]] : f32
// CHECK:         %[[VAL5:.*]] = affine.load %[[ARG1]][%[[ARG5]]] : memref<?xf32>
// CHECK:         %[[VAL6:.*]] = mulf %[[VAL5]], %[[VAL4]] : f32
// CHECK:         affine.store %[[VAL6]], %[[ARG3]][%[[ARG5]]] : memref<?xf32>
// CHECK:       }
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
