// RUN: polymer-opt %s -array-expansion -split-input-file | FileCheck %s

func @scratchpad_in_single_loop(%A: memref<?xf32>, %B: memref<?xf32>) {
  %c0 = constant 0 : index
  %N = memref.dim %A, %c0 : memref<?xf32>

  %scr = memref.alloca() {scop.scratchpad}: memref<1xf32>
  affine.for %i = 0 to %N {
    %0 = affine.load %A[%i] : memref<?xf32>
    affine.store %0, %scr[0] : memref<1xf32>
    %1 = addf %0, %0 : f32
    affine.store %1, %A[%i] : memref<?xf32>
    %2 = affine.load %scr[0] : memref<1xf32>
    %3 = mulf %2, %2 : f32
    affine.store %3, %B[%i] : memref<?xf32>
  }

  return
}

// CHECK: #[[MAP:.*]] = affine_map<()[s0] -> (s0)>
// CHECK: func @scratchpad_in_single_loop(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>) {
// CHECK-NEXT:   %[[C0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[VAL0:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?xf32>
// CHECK-NEXT:   %[[VAL1:.*]] = affine.max #map()[%[[VAL0]]]
// CHECK-NEXT:   %[[VAL2:.*]] = memref.alloca(%[[VAL1]]) {scop.scratchpad} : memref<?xf32>
// CHECK-NEXT:   affine.for %[[ARG2:.*]] = 0 to %[[VAL0]] {
// CHECK-NEXT:     %[[VAL3:.*]] = affine.load %[[ARG0]][%[[ARG2]]] : memref<?xf32>
// CHECK-NEXT:     affine.store %[[VAL3]], %[[VAL2]][%[[ARG2]]] : memref<?xf32>
// CHECK-NEXT:     %[[VAL4:.*]] = addf %[[VAL3]], %[[VAL3]] : f32
// CHECK-NEXT:     affine.store %[[VAL4]], %[[ARG0]][%[[ARG2]]] : memref<?xf32>
// CHECK-NEXT:     %[[VAL5:.*]] = affine.load %[[VAL2]][%[[ARG2]]] : memref<?xf32>
// CHECK-NEXT:     %[[VAL6:.*]] = mulf %[[VAL5]], %[[VAL5]] : f32
// CHECK-NEXT:     affine.store %[[VAL6]], %[[ARG1]][%[[ARG2]]] : memref<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

