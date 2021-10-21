// RUN: polymer-opt %s -reg2mem | FileCheck %s
func @foo(%A: memref<?xf32>) attributes {scop.ignored} {
  %0 = affine.load %A[0] : memref<?xf32>
  affine.for %i = 1 to 30 {
    affine.store %0, %A[%i] : memref<?xf32>
  }

  return
}

// CHECK: func @foo
// CHECK-NEXT:   %[[v0:.*]] = affine.load 
// CHECK-NEXT:   affine.for %{{.*}} = 1 to 30 
// CHECK-NEXT:     affine.store %[[v0]]
