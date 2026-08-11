// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cublasSgemm_tn(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>,
                              %c: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %c : tensor<?x?xf32>
  }
  kernel.defn @cublasSaxpby(%x: tensor<?xf32>, %y: tensor<?xf32>,
                            %a: f32, %b: f32) -> tensor<?xf32> {
    kernel.yield %y : tensor<?xf32>
  }

  func.func @gemm_tn(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>,
                     %c: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %r = kernel.launch @cublasSgemm_tn(%a, %b, %c)
        : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }

  func.func @axpby(%x: tensor<?xf32>, %y: tensor<?xf32>,
                   %a: f32, %b: f32) -> tensor<?xf32> {
    %r = kernel.launch @cublasSaxpby(%x, %y, %a, %b)
        : (tensor<?xf32>, tensor<?xf32>, f32, f32) -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @gemm_tn
// CHECK: %[[TA:.*]] = arith.constant 1 : i32
// CHECK: %[[TB:.*]] = arith.constant 0 : i32
// CHECK: call @polygeist_cublas_sgemm_transpose
// CHECK-SAME: %[[TA]], %[[TB]]
// CHECK-NOT: kernel.launch

// CHECK-LABEL: func.func @axpby
// CHECK: call @polygeist_cublas_saxpby
// CHECK-NOT: kernel.launch
