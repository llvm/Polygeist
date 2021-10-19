// RUN: polymer-opt %s -reg2mem | FileCheck %s

// This is a general test case that covers many different aspects for checking.

func @load_store_dep(%A: memref<?xf32>, %B: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %someValue = arith.constant 1.23 : f32

  %NI = memref.dim %A, %c0 : memref<?xf32>
  %NJ = memref.dim %B, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %NI {
    %0 = affine.load %A[%i] : memref<?xf32>
    %1 = arith.mulf %0, %0 : f32
    affine.store %someValue, %A[%i] : memref<?xf32>

    affine.for %j = 0 to %NJ {
      %2 = arith.mulf %1, %0 : f32
      %3 = arith.addf %1, %2 : f32
      %4 = arith.subf %3, %someValue : f32
      affine.store %4, %B[%i, %j] : memref<?x?xf32>
    }

    affine.store %1, %A[%i] : memref<?xf32>
  }

  return 
}

// CHECK:       func @load_store_dep(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?x?xf32>) {
// CHECK-NEXT:   %[[MEM0:.*]] = memref.alloca() {scop.scratchpad} : memref<1xf32>
// CHECK-NEXT:   %[[MEM1:.*]] = memref.alloca() {scop.scratchpad} : memref<1xf32>
// CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[CST:.*]] = arith.constant 1.230000e+00 : f32
// CHECK-NEXT:   %[[VAL2:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?xf32>
// CHECK-NEXT:   %[[VAL3:.*]] = memref.dim %[[ARG1]], %[[C1]] : memref<?x?xf32>
// CHECK-NEXT:   affine.for %[[ARG2:.*]] = 0 to %[[VAL2]] {
// CHECK-NEXT:     %[[VAL4:.*]] = affine.load %[[ARG0]][%[[ARG2]]] : memref<?xf32>
// CHECK-NEXT:     affine.store %[[VAL4]], %[[MEM0]][0] : memref<1xf32>
// CHECK-NEXT:     %[[VAL5:.*]] = arith.mulf %[[VAL4]], %[[VAL4]] : f32
// CHECK-NEXT:     affine.store %[[VAL5]], %[[MEM1]][0] : memref<1xf32>
// CHECK-NEXT:     affine.store %[[CST]], %[[ARG0]][%[[ARG2]]] : memref<?xf32>
// CHECK-NEXT:     affine.for %[[ARG3:.*]] = 0 to %[[VAL3]] {
// CHECK-NEXT:       %[[VAL6:.*]] = affine.load %[[MEM0]][0] : memref<1xf32>
// CHECK-NEXT:       %[[VAL7:.*]] = affine.load %[[MEM1]][0] : memref<1xf32>
// CHECK-NEXT:       %[[VAL8:.*]] = arith.mulf %[[VAL7]], %[[VAL6]] : f32
// CHECK-NEXT:       %[[VAL9:.*]] = arith.addf %[[VAL7]], %[[VAL8]] : f32
// CHECK-NEXT:       %[[VAL10:.*]] = arith.subf %[[VAL9]], %[[CST]] : f32
// CHECK-NEXT:       affine.store %[[VAL10]], %[[ARG1]][%[[ARG2]], %[[ARG3]]] : memref<?x?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %[[VAL5]], %[[ARG0]][%[[ARG2]]] : memref<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

