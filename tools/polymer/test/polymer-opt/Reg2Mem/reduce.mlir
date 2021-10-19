// RUN: polymer-opt %s -demote-loop-reduction -reg2mem -split-input-file | FileCheck %s

func @reduce_with_iter_args(%A: memref<?xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %N = memref.dim %A, %c0 : memref<?xf32>

  %sum_0 = arith.constant 0.0 : f32
  %prod_0 = arith.constant 1.0 : f32

  %sum, %prod = affine.for %i = 0 to %N iter_args(%sum_iter=%sum_0, %prod_iter=%prod_0) -> (f32, f32) {
    %0 = affine.load %A[%i] : memref<?xf32>
    %1 = arith.addf %sum_iter, %0 : f32
    %2 = arith.mulf %prod_iter, %0 : f32
    affine.yield %1, %2 : f32, f32
  }

  %ans = arith.addf %sum, %prod : f32

  return %ans : f32
}

// CHECK:      func @reduce_with_iter_args(%[[ARG0:.*]]: memref<?xf32>) -> f32 {
// CHECK-NEXT:   %[[CST0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[VAL0:.*]] = memref.dim %[[ARG0]], %[[CST0]] : memref<?xf32>
// CHECK-NEXT:   %[[CST1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %[[CST2:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:   %[[MEM1:.*]] = memref.alloca() {scop.scratchpad} : memref<1xf32>
// CHECK-NEXT:   affine.store %[[CST1]], %[[MEM1]][0] : memref<1xf32>
// CHECK-NEXT:   %[[MEM2:.*]] = memref.alloca() {scop.scratchpad} : memref<1xf32>
// CHECK-NEXT:   affine.store %[[CST2]], %[[MEM2]][0] : memref<1xf32>
// CHECK-NEXT:   affine.for %[[ARG1:.*]] = 0 to %[[VAL0]] {
// CHECK-NEXT:     %[[VAL6:.*]] = affine.load %[[MEM2]][0] : memref<1xf32>
// CHECK-NEXT:     %[[VAL7:.*]] = affine.load %[[MEM1]][0] : memref<1xf32>
// CHECK-NEXT:     %[[VAL8:.*]] = affine.load %[[ARG0]][%[[ARG1]]] : memref<?xf32>
// CHECK-NEXT:     %[[VAL9:.*]] = arith.addf %[[VAL7]], %[[VAL8]] : f32
// CHECK-NEXT:     %[[VAL10:.*]] = arith.mulf %[[VAL6]], %[[VAL8]] : f32
// CHECK-NEXT:     affine.store %[[VAL9]], %[[MEM1]][0] : memref<1xf32>
// CHECK-NEXT:     affine.store %[[VAL10]], %[[MEM2]][0] : memref<1xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[VAL3:.*]] = affine.load %[[MEM2]][0] : memref<1xf32>
// CHECK-NEXT:   %[[VAL4:.*]] = affine.load %[[MEM1]][0] : memref<1xf32>
// CHECK-NEXT:   %[[VAL5:.*]] = arith.addf %[[VAL4]], %[[VAL3]] : f32
// CHECK-NEXT:   return %[[VAL5]] : f32
// CHECK-NEXT: }

// -----

func @nested(%A: memref<?xf32>, %B: memref<?x?xf32>, %out: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %N = memref.dim %A, %c0 : memref<?xf32>
  %M = memref.dim %B, %c0 : memref<?x?xf32>
  %K = memref.dim %B, %c1 : memref<?x?xf32>

  %sum_0 = arith.constant 0.0 : f32
  %sum_a = affine.for %i = 0 to %N iter_args(%sum_iter=%sum_0) -> (f32) {
    %0 = affine.load %A[%i] : memref<?xf32>
    %1 = arith.addf %sum_iter, %0 : f32
    affine.yield %1 : f32
  }

  %sum_b = affine.for %i = 0 to %M iter_args(%sum_iter=%sum_a) -> (f32) {
    %sum_b_inner = affine.for %j = 0 to %K iter_args(%sum_iter_inner=%sum_iter) -> (f32) {
      %0 = affine.load %B[%i,%j] : memref<?x?xf32>
      %1 = arith.addf %sum_iter_inner, %0 : f32
      affine.yield %1 : f32
    }
    affine.yield %sum_b_inner : f32
  }

  affine.store %sum_b, %out[0] : memref<1xf32>

  return
}

// CHECK: func @nested(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<1xf32>) {
// CHECK:   %[[MEM0:.*]] = memref.alloca() {scop.scratchpad} : memref<1xf32>
// CHECK:   affine.store %{{.*}}, %[[MEM0]][0] : memref<1xf32>
// CHECK:   affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK:     %{{.*}} = affine.load %[[MEM0]][0] : memref<1xf32>
// CHECK:     affine.store %{{.*}}, %[[MEM0]][0] : memref<1xf32>
// CHECK:   }
// CHECK:   %[[VAL0:.*]] = affine.load %[[MEM0]][0] : memref<1xf32>
// CHECK:   %[[MEM1:.*]] = memref.alloca() {scop.scratchpad} : memref<1xf32>
// CHECK:   affine.store %[[VAL0]], %[[MEM1]][0] : memref<1xf32>
// CHECK:   affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK:     %{{.*}} = affine.load %[[MEM1]][0] : memref<1xf32>
// CHECK:     %[[MEM2:.*]] = memref.alloca() {scop.scratchpad} : memref<1xf32>
// CHECK:     affine.store %{{.*}}, %[[MEM2]][0] : memref<1xf32>
// CHECK:     affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK:       %{{.*}} = affine.load %[[MEM2]][0] : memref<1xf32>
// CHECK:       affine.store %{{.*}}, %[[MEM2]][0] : memref<1xf32>
// CHECK:     }
// CHECK:     %{{.*}} = affine.load %[[MEM2]][0] : memref<1xf32>
// CHECK:     affine.store %{{.*}}, %[[MEM1]][0] : memref<1xf32>
// CHECK:   }
// CHECK:   %{{.*}} = affine.load %[[MEM1]][0] : memref<1xf32>
// CHECK:   affine.store %{{.*}}, %[[ARG2]][0] : memref<1xf32>
// CHECK:   return
// CHECK: }
