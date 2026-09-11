// RUN: polygeist-opt --raise-affine-to-linalg-pipeline %s | FileCheck %s

// Regression for non-termination caused by repeatedly moving payload
// arithmetic while raising a dynamic GEMM-shaped nest. Only index-valued
// loop-domain arithmetic is eligible for the pre-raise invariant hoist.
module {
  func.func @dynamic_gemm(%m: index, %n: index, %k: index, %alpha: f64,
                          %A: memref<?x?xf64>, %B: memref<?x?xf64>,
                          %C: memref<?x?xf64>) {
    affine.for %i = 0 to %m {
      affine.for %j = 0 to %n {
        affine.for %r = 0 to %k {
          %a = affine.load %A[%i, %r] : memref<?x?xf64>
          %scaled = arith.mulf %alpha, %a : f64
          %b = affine.load %B[%r, %j] : memref<?x?xf64>
          %product = arith.mulf %scaled, %b : f64
          %old = affine.load %C[%i, %j] : memref<?x?xf64>
          %new = arith.addf %old, %product : f64
          affine.store %new, %C[%i, %j] : memref<?x?xf64>
        }
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @dynamic_gemm
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.yield
