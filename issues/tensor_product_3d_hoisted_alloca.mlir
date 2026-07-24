module {
  func.func @tensor_product_3d_hoisted_alloca(%psi: memref<?xf32>,
                                              %u: memref<?xf32>,
                                              %out: memref<?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %qi = 0 to 5 {
      affine.for %qj = 0 to 5 {
        affine.for %qk = 0 to 5 {
          affine.store %cst, %out[%qk + %qi * 25 + %qj * 5] : memref<?xf32>
          %slot_j = memref.alloca() : memref<f32>
          %slot_k = memref.alloca() : memref<f32>
          affine.for %i = 0 to 4 {
            %out_old = affine.load %out[%qk + %qi * 25 + %qj * 5] : memref<?xf32>
            %psi_i = affine.load %psi[%i + %qi * 4] : memref<?xf32>
            affine.store %out_old, %slot_j[] : memref<f32>
            affine.for %j = 0 to 4 {
              %acc_j = affine.load %slot_j[] : memref<f32>
              %psi_j = affine.load %psi[%j + %qj * 4] : memref<?xf32>
              %partial_ij = arith.mulf %psi_i, %psi_j : f32
              affine.store %acc_j, %slot_k[] : memref<f32>
              affine.for %k = 0 to 4 {
                %acc_k = affine.load %slot_k[] : memref<f32>
                %psi_k = affine.load %psi[%k + %qk * 4] : memref<?xf32>
                %u_ijk = affine.load %u[%k + %i * 16 + %j * 4] : memref<?xf32>
                %term0 = arith.mulf %partial_ij, %psi_k : f32
                %term1 = arith.mulf %term0, %u_ijk : f32
                %next = arith.addf %acc_k, %term1 : f32
                affine.store %next, %slot_k[] : memref<f32>
              }
              %after_k = affine.load %slot_k[] : memref<f32>
              affine.store %after_k, %slot_j[] : memref<f32>
            }
            %after_j = affine.load %slot_j[] : memref<f32>
            affine.store %after_j, %out[%qk + %qi * 25 + %qj * 5] : memref<?xf32>
          }
        }
      }
    }
    return
  }
}
