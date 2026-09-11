// RUN: polygeist-opt --raise-affine-to-linalg %s | FileCheck %s

module {
  func.func @ikj_promotes_scalar_load(%A: memref<8x3xf32>,
                                      %B: memref<3x16xf32>,
                                      %C: memref<8x16xf32>) {
    %alpha = arith.constant 1.000000e+00 : f32
    affine.for %i = 0 to 8 {
      affine.for %k = 0 to 3 {
        %a = affine.load %A[%i, %k] : memref<8x3xf32>
        %a_part = arith.mulf %alpha, %a : f32
        affine.for %j = 0 to 16 {
          %b = affine.load %B[%k, %j] : memref<3x16xf32>
          %c = affine.load %C[%i, %j] : memref<8x16xf32>
          %mul = arith.mulf %a_part, %b : f32
          %sum = arith.addf %c, %mul : f32
          affine.store %sum, %C[%i, %j] : memref<8x16xf32>
        }
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @ikj_promotes_scalar_load
// CHECK-NOT: affine.for
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK: arith.mulf
// CHECK: linalg.yield
// CHECK-NOT: affine.for
// CHECK: return
