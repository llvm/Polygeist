// RUN: polymer-translate %s -mlir-to-openscop | FileCheck %s

// Consider if operations in the domain.
// We will make this test valid when the diff D86421 is landed.

#set = affine_set<(d0, d1): (d0 - 16 >= 0, d1 - 16 >= 0)>

func @load_store_if(%A : memref<32x32xf32>, %B : memref<32x32xf32>) -> () {
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      affine.if #set(%i, %j) {
        %0 = affine.load %A[%i, %j] : memref<32x32xf32>
        affine.store %0, %A[%i, %j] : memref<32x32xf32>
      }
    }
  }

  return
}

// CHECK: <OpenScop>
