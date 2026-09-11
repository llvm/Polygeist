// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cublasSgemm_tn(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>,
                              %c: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %c : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nt_alpha_beta(
      %a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>,
      %beta: f32, %alpha: f32) -> tensor<?x?xf32> {
    kernel.yield %c : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tt_alpha(
      %a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>,
      %alpha: f32) -> tensor<?x?xf32> {
    kernel.yield %c : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nt_zero(
      %a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>)
      -> tensor<?x?xf32> {
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

  func.func @gemm_nt_alpha_beta(
      %a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>,
      %beta: f32, %alpha: f32) -> tensor<?x?xf32> {
    %r = kernel.launch @cublasSgemm_nt_alpha_beta(
        %a, %b, %c, %beta, %alpha)
        : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, f32, f32)
          -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }

  func.func @gemm_tt_alpha(
      %a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>,
      %alpha: f32) -> tensor<?x?xf32> {
    %r = kernel.launch @cublasSgemm_tt_alpha(%a, %b, %c, %alpha)
        : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, f32)
          -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }

  func.func @gemm_nt_zero(
      %a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>)
      -> tensor<?x?xf32> {
    %r = kernel.launch @cublasSgemm_nt_zero(%a, %b, %c)
        : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
          -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
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

// CHECK-LABEL: func.func @gemm_nt_alpha_beta
// CHECK: %[[NTA:.*]] = arith.constant 0 : i32
// CHECK: %[[NTB:.*]] = arith.constant 1 : i32
// CHECK: call @polygeist_cublas_sgemm_transpose
// CHECK-SAME: %[[NTA]], %[[NTB]], %arg4
// CHECK-SAME: %arg3
// CHECK-NOT: kernel.launch

// CHECK-LABEL: func.func @gemm_tt_alpha
// CHECK: %[[TTA:.*]] = arith.constant 1 : i32
// CHECK: %[[TTB:.*]] = arith.constant 1 : i32
// CHECK: call @polygeist_cublas_sgemm_transpose
// CHECK-SAME: %[[TTA]], %[[TTB]], %arg3
// CHECK-NOT: kernel.launch

// CHECK-LABEL: func.func @gemm_nt_zero
// CHECK: %[[ZNTA:.*]] = arith.constant 0 : i32
// CHECK: %[[ZNTB:.*]] = arith.constant 1 : i32
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: call @polygeist_cublas_sgemm_transpose
// CHECK-SAME: %[[ZNTA]], %[[ZNTB]]
// CHECK-SAME: %[[ZERO]]
// CHECK-NOT: kernel.launch
