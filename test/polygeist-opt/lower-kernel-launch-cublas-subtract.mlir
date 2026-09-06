// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s

module {
  kernel.defn @cublasDgemv_subtract_T(
      %a: tensor<?x?xf64>, %x: tensor<?xf64>, %y: tensor<?xf64>)
      -> tensor<?xf64> { kernel.yield %y : tensor<?xf64> }
  kernel.defn @cublasDgemm_subtract(
      %a: tensor<?x?xf64>, %b: tensor<?x?xf64>, %c: tensor<?x?xf64>)
      -> tensor<?x?xf64> { kernel.yield %c : tensor<?x?xf64> }

  func.func @subtract_updates(%a: tensor<?x?xf64>, %x: tensor<?xf64>,
                              %y: tensor<?xf64>, %b: tensor<?x?xf64>,
                              %c: tensor<?x?xf64>)
      -> (tensor<?xf64>, tensor<?x?xf64>) {
    %yv = kernel.launch @cublasDgemv_subtract_T(%a, %x, %y) :
        (tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    %cv = kernel.launch @cublasDgemm_subtract(%a, %b, %c) :
        (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>)
        -> tensor<?x?xf64>
    return %yv, %cv : tensor<?xf64>, tensor<?x?xf64>
  }
}

// CHECK-LABEL: func.func @subtract_updates
// CHECK: %[[NEG_GEMV:.*]] = arith.constant -1.000000e+00 : f64
// CHECK: call @polygeist_cublas_dgemv_T({{.*}}, %[[NEG_GEMV]],
// CHECK: %[[NEG_GEMM:.*]] = arith.constant -1.000000e+00 : f64
// CHECK: call @polygeist_cublas_dgemm({{.*}}, %[[NEG_GEMM]],
// CHECK-NOT: kernel.launch
