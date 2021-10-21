// RUN: polymer-opt %s -fold-scf-if | FileCheck %s

func @foo(%A: memref<10xf32>, %a: f32, %cond: i1) {
  scf.if %cond {
    affine.store %a, %A[0] : memref<10xf32>
  }
  return
}

// CHECK: func @foo(%[[A:.*]]: memref<10xf32>, %[[a:.*]]: f32, %[[cond:.*]]: i1) 
// CHECK-NEXT:   %[[v0:.*]] = affine.load %[[A]][0] : memref<10xf32>
// CHECK-NEXT:   %[[v1:.*]] = select %[[cond]], %[[a]], %[[v0]] : f32
// CHECK-NEXT:   affine.store %[[v1]], %[[A]][0] : memref<10xf32>
