// RUN: polymer-opt %s -fold-scf-if | FileCheck %s

func @foo(%A: memref<10xf32>, %B: memref<10xf32>, %a: f32, %b: f32, %cond: i1) {
  scf.if %cond {
    affine.store %a, %A[0] : memref<10xf32>
  } else {
    affine.store %b, %B[0] : memref<10xf32>
  }
  return
}

// CHECK: func @foo(%[[A:.*]]: memref<10xf32>, %[[B:.*]]: memref<10xf32>, %[[a:.*]]: f32, %[[b:.*]]: f32, %[[cond:.*]]: i1) 
// CHECK-NEXT:   %[[v0:.*]] = affine.load %[[B]][0] : memref<10xf32>
// CHECK-NEXT:   %[[v1:.*]] = affine.load %[[A]][0] : memref<10xf32>
// CHECK-NEXT:   %[[v2:.*]] = arith.select %[[cond]], %[[a]], %[[v1]] : f32
// CHECK-NEXT:   %[[v3:.*]] = arith.select %[[cond]], %[[v0]], %[[b]] : f32
// CHECK-NEXT:   affine.store %[[v2]], %[[A]][0] : memref<10xf32>
// CHECK-NEXT:   affine.store %[[v3]], %[[B]][0] : memref<10xf32>
