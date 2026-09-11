// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cublasDgemv_T_zero(
      %a: tensor<?x?xf64>, %x: tensor<?xf64>, %out: tensor<?xf64>)
      -> tensor<?xf64> {
    kernel.yield %out : tensor<?xf64>
  }
  func.func @overwrite(%a: tensor<?x?xf64>, %x: tensor<?xf64>,
                       %out: tensor<?xf64>) -> tensor<?xf64> {
    %r = kernel.launch @cublasDgemv_T_zero(%a, %x, %out)
        : (tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    return %r : tensor<?xf64>
  }
}

// CHECK-LABEL: func.func @overwrite
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: call @polygeist_cublas_dgemv_T
// CHECK-SAME: %[[ZERO]]
// CHECK: return %{{.*}} : tensor<?xf64>
// CHECK-NOT: kernel.launch
