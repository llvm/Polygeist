// RUN: polymer-opt %s -extract-scop-stmt | FileCheck %s

func @foo(%A: memref<1024xi32>, %B: memref<1024x16xi32>) {
  %c4_i32 = arith.constant 4 : i32
  %c15_i32 = arith.constant 15 : i32

  affine.for %i = 1 to 5 {
    affine.for %j = 0 to 16 {
      %1 = affine.load %A[%j * 4] : memref<1024xi32>
      %2 = arith.shrsi %1, %c4_i32 : i32
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.andi %1, %c15_i32 : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = memref.load %B[%3, %5] : memref<1024x16xi32>
      affine.store %6, %A[%j * 4] : memref<1024xi32>
    }
  }

  return
}

// CHECK: func private @S0
// CHECK: %{{.*}} = affine.load
// CHECK-NEXT: %{{.*}} = arith.shrsi
// CHECK-NEXT: %{{.*}} = arith.index_cast
// CHECK-NEXT: %{{.*}} = arith.andi
// CHECK-NEXT: %{{.*}} = arith.index_cast
// CHECK-NEXT: %{{.*}} = memref.load
// CHECK-NEXT: affine.store

// CHECK: func @foo(%[[ARG0:.*]]: memref<1024xi32>, %[[ARG1:.*]]: memref<1024x16xi32>)
// CHECK-NEXT: affine.for %[[I:.*]] = 1 to 5
// CHECK-NEXT: affine.for %[[J:.*]] = 0 to 16
// CHECK-NEXT: call @S0(%[[ARG0]], %[[J]], %[[ARG1]])
