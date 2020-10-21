// RUN: polymer-opt %s -pluto-opt | FileCheck %s

func @matmul() {
  %A = alloc() : memref<64x64xf32>
  %B = alloc() : memref<64x64xf32>
  %C = alloc() : memref<64x64xf32>

  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %0 = affine.load %A[%i, %k] : memref<64x64xf32>
        %1 = affine.load %B[%k, %j] : memref<64x64xf32>
        %2 = mulf %0, %1 : f32
        %3 = affine.load %C[%i, %j] : memref<64x64xf32>
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<64x64xf32>
      }
    }
  }

  return
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0) -> (d0 * 32 + 31)>
// CHECK-DAG: #[[MAP3:.*]] = affine_map<() -> (0)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<() -> (1)>
//
//
// CHECK: module {
// CHECK:   func @main(%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?x?xf32>) {
// CHECK:     affine.for %[[ARG3:.*]] = 0 to 1 {
// CHECK:       affine.for %[[ARG4:.*]] = 0 to 1 {
// CHECK:         affine.for %[[ARG5:.*]] = 0 to 1 {
// CHECK:           affine.for %[[ARG6:.*]] = #[[MAP1]](%[[ARG3]]) to #[[MAP2]](%[[ARG3]]) {
// CHECK:             affine.for %[[ARG7:.*]] = #[[MAP1]](%[[ARG5]]) to #[[MAP2]](%[[ARG5]]) {
// CHECK:               affine.for %[[ARG8:.*]] = #[[MAP1]](%[[ARG4]]) to #[[MAP2]](%[[ARG4]]) {
// CHECK:                 %[[VAL0:.*]] = affine.load %[[ARG0]][%[[ARG6]], %[[ARG8]]] : memref<?x?xf32>
// CHECK:                 %[[VAL1:.*]] = affine.load %[[ARG2]][%[[ARG7]], %[[ARG8]]] : memref<?x?xf32>
// CHECK:                 %[[VAL2:.*]] = affine.load %[[ARG1]][%[[ARG6]], %[[ARG7]]] : memref<?x?xf32>
// CHECK:                 %[[VAL3:.*]] = mulf %[[VAL2]], %[[VAL1]] : f32
// CHECK:                 %[[VAL4:.*]] = addf %[[VAL3]], %[[VAL0]] : f32
// CHECK:                 affine.store %[[VAL4]], %[[ARG0]][%[[ARG6]], %[[ARG8]]] : memref<?x?xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
