// RUN: polymer-opt %s -pluto-opt | FileCheck %s


func @S0(%A: memref<10x60xf32>, %i: index, %j: index) attributes {scop.stmt} {
  %cst = arith.constant 3.14 : f32
  affine.store %cst, %A[%i, %j] : memref<10x60xf32>
  return
}

func @foo(%A: memref<10x60xf32>, %i: index) {
  affine.for %j = 0 to 60 {
    call @S0(%A, %i, %j): (memref<10x60xf32>, index, index) -> ()
  }
  return
}

// CHECK: func @foo(%[[A:.*]]: memref<{{.*}}>, %[[i:.*]]: index)
// CHECK-NEXT: affine.for 
// CHECK-NEXT: affine.for %[[j:.*]] =
// CHECK-NEXT: call @S0(%[[A]], %[[i]], %[[j]])
