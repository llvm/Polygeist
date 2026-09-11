// RUN: polygeist-opt --remove-iter-args --affine-parallelize --raise-affine-to-linalg-pipeline --lower-polygeist-submap %s | FileCheck %s

module {
  func.func @tensor_product_3d(%psi: memref<?xf32>, %u: memref<?xf32>,
                               %out: memref<?xf32>) {
    %zero = arith.constant 0.0 : f32
    affine.for %qi = 0 to 5 {
      affine.for %qj = 0 to 5 {
        affine.for %qk = 0 to 5 {
          %sum_i = affine.for %i = 0 to 4
              iter_args(%acc_i = %zero) -> (f32) {
            %psi_i = affine.load %psi[%i + %qi * 4] : memref<?xf32>
            %sum_j = affine.for %j = 0 to 4
                iter_args(%acc_j = %acc_i) -> (f32) {
              %psi_j = affine.load %psi[%j + %qj * 4] : memref<?xf32>
              %partial = arith.mulf %psi_i, %psi_j : f32
              %sum_k = affine.for %k = 0 to 4
                  iter_args(%acc_k = %acc_j) -> (f32) {
                %psi_k = affine.load %psi[%k + %qk * 4] : memref<?xf32>
                %u_ijk = affine.load %u[%k + %i * 16 + %j * 4]
                    : memref<?xf32>
                %term0 = arith.mulf %partial, %psi_k : f32
                %term1 = arith.mulf %term0, %u_ijk : f32
                %next = arith.addf %acc_k, %term1 : f32
                affine.yield %next : f32
              }
              affine.yield %sum_k : f32
            }
            affine.yield %sum_j : f32
          }
          affine.store %sum_i, %out[%qk + %qi * 25 + %qj * 5]
              : memref<?xf32>
        }
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @tensor_product_3d
// CHECK-NOT: memref.alloca
// CHECK-NOT: affine.for
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// CHECK-NOT: affine.for
// CHECK-NOT: memref.alloca
