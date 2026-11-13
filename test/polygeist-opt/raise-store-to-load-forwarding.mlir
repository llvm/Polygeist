// RUN: polygeist-opt %s --remove-iter-args --affine-parallelize \
// RUN:   --raise-affine-to-linalg-pipeline | FileCheck %s

module {
  // A later load from r[i] must observe the value stored earlier in the same
  // iteration.  Making that load another linalg input would instead expose
  // the pre-iteration value and silently corrupt the dot-product reduction.
  func.func @pcg_step(%ap: memref<32xf64>,
                      %inv_diag: memref<32xf64>,
                      %x: memref<32xf64>,
                      %r: memref<32xf64>,
                      %z: memref<32xf64>,
                      %alpha: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %sum = affine.for %i = 0 to 32 iter_args(%acc = %zero) -> f64 {
      %ap_i = affine.load %ap[%i] : memref<32xf64>
      %scaled = arith.mulf %alpha, %ap_i : f64
      %old_r = affine.load %r[%i] : memref<32xf64>
      %new_r = arith.subf %old_r, %scaled : f64
      affine.store %new_r, %r[%i] : memref<32xf64>
      %diag = affine.load %inv_diag[%i] : memref<32xf64>
      %new_z = arith.mulf %diag, %new_r : f64
      affine.store %new_z, %z[%i] : memref<32xf64>
      %reloaded_r = affine.load %r[%i] : memref<32xf64>
      %product = arith.mulf %reloaded_r, %new_z : f64
      %next = arith.addf %acc, %product : f64
      affine.yield %next : f64
    }
    return %sum : f64
  }
}

// CHECK-LABEL: func.func @pcg_step
// CHECK: linalg.generic
// CHECK: ^bb0(%[[AP:[A-Za-z0-9_]+]]: f64, %[[DIAG:[A-Za-z0-9_]+]]: f64, %[[OLD_R:[A-Za-z0-9_]+]]: f64, %[[OLD_Z:[A-Za-z0-9_]+]]: f64, %[[ACC:[A-Za-z0-9_]+]]: f64):
// CHECK: %[[NEW_R:.*]] = arith.subf %[[OLD_R]],
// CHECK: %[[NEW_Z:.*]] = arith.mulf {{.*}}, %[[NEW_R]]
// CHECK: %[[PRODUCT:.*]] = arith.mulf %[[NEW_R]], %[[NEW_Z]]
// CHECK: arith.addf %[[ACC]], %[[PRODUCT]]
