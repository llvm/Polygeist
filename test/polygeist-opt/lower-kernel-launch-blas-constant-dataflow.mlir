// RUN: polygeist-opt --split-input-file --lower-kernel-launch-to-cublas %s | FileCheck %s

// A zero produced by one matched candidate remains visible through a tensor
// slice when the following ordinary GEMV candidate is lowered.  The semantic
// match does not need a cublasDgemv_T_zero symbol to select beta=0.
module {
  kernel.defn @memset_zero_1D(%out: tensor<?xf64>) -> tensor<?xf64> {
    kernel.yield %out : tensor<?xf64>
  }
  kernel.defn @cublasDgemv(
      %a: tensor<?x?xf64>, %x: tensor<?xf64>, %out: tensor<?xf64>)
      -> tensor<?xf64> {
    kernel.yield %out : tensor<?xf64>
  }

  func.func @zero_through_slice(
      %a: tensor<?x?xf64>, %x: tensor<?xf64>, %out: tensor<?xf64>,
      %count: index) -> tensor<?xf64> {
    %zeroed = kernel.launch @memset_zero_1D(%out)
        : (tensor<?xf64>) -> tensor<?xf64>
    %slice = tensor.extract_slice %zeroed[0] [%count] [1]
        : tensor<?xf64> to tensor<?xf64>
    %result = kernel.launch @cublasDgemv(%a, %x, %slice)
        : (tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    return %result : tensor<?xf64>
  }
}

// CHECK-LABEL: func.func @zero_through_slice
// CHECK: call @polygeist_cublas_memset_zero_1d
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: call @polygeist_cublas_dgemv
// CHECK-SAME: %[[ZERO]]
// CHECK-NOT: kernel.launch

// -----

// The analysis is datatype-independent: the same whole-value proof selects
// beta=0 for an ordinary FP32 SGEMM match and removes its dead fill.
module {
  kernel.defn @memset_zero_2D_f32(%out: tensor<?x?xf32>)
      -> tensor<?x?xf32> {
    kernel.yield %out : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tn(
      %a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %out: tensor<?x?xf32>)
      -> tensor<?x?xf32> {
    kernel.yield %out : tensor<?x?xf32>
  }

  func.func @zero_sgemm(
      %a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %out: tensor<?x?xf32>)
      -> tensor<?x?xf32> {
    %zeroed = kernel.launch @memset_zero_2D_f32(%out)
        : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %result = kernel.launch @cublasSgemm_tn(%a, %b, %zeroed)
        : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
          -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @zero_sgemm
// CHECK-NOT: polygeist_cublas_memset_zero
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: call @polygeist_cublas_sgemm_transpose
// CHECK-SAME: %[[ZERO]]
// CHECK-NOT: kernel.launch

// -----

// A whole-value zero fill with no other user is removed after beta=0 makes
// the following ordinary GEMV an overwrite.
module {
  kernel.defn @memset_zero_1D(%out: tensor<?xf64>) -> tensor<?xf64> {
    kernel.yield %out : tensor<?xf64>
  }
  kernel.defn @cublasDgemv(
      %a: tensor<?x?xf64>, %x: tensor<?xf64>, %out: tensor<?xf64>)
      -> tensor<?xf64> {
    kernel.yield %out : tensor<?xf64>
  }

  func.func @dead_whole_fill(
      %a: tensor<?x?xf64>, %x: tensor<?xf64>, %out: tensor<?xf64>)
      -> tensor<?xf64> {
    %zeroed = kernel.launch @memset_zero_1D(%out)
        : (tensor<?xf64>) -> tensor<?xf64>
    %result = kernel.launch @cublasDgemv(%a, %x, %zeroed)
        : (tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    return %result : tensor<?xf64>
  }
}

// CHECK-LABEL: func.func @dead_whole_fill
// CHECK-NOT: polygeist_cublas_memset_zero
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: call @polygeist_cublas_dgemv
// CHECK-SAME: %[[ZERO]]
// CHECK-NOT: kernel.launch

// -----

// An unknown incoming destination must retain the ordinary accumulate form.
module {
  kernel.defn @cublasDgemv(
      %a: tensor<?x?xf64>, %x: tensor<?xf64>, %out: tensor<?xf64>)
      -> tensor<?xf64> {
    kernel.yield %out : tensor<?xf64>
  }

  func.func @unknown_destination(
      %a: tensor<?x?xf64>, %x: tensor<?xf64>, %out: tensor<?xf64>)
      -> tensor<?xf64> {
    %result = kernel.launch @cublasDgemv(%a, %x, %out)
        : (tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    return %result : tensor<?xf64>
  }
}

// CHECK-LABEL: func.func @unknown_destination
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f64
// CHECK: call @polygeist_cublas_dgemv
// CHECK-SAME: %[[ONE]]
// CHECK-NOT: kernel.launch

// -----

// A dynamic GEMM beta may be NaN.  Even with a zero destination, replacing
// beta*0 by 0 would then change IEEE semantics, so no propagation is allowed.
module {
  kernel.defn @memset_zero_2D(%out: tensor<?x?xf64>) -> tensor<?x?xf64> {
    kernel.yield %out : tensor<?x?xf64>
  }
  kernel.defn @cublasDgemm(
      %a: tensor<?x?xf64>, %b: tensor<?x?xf64>, %out: tensor<?x?xf64>,
      %beta: f64, %alpha: f64) -> tensor<?x?xf64> {
    kernel.yield %out : tensor<?x?xf64>
  }

  func.func @dynamic_beta(
      %a: tensor<?x?xf64>, %b: tensor<?x?xf64>, %out: tensor<?x?xf64>,
      %beta: f64, %alpha: f64) -> tensor<?x?xf64> {
    %zeroed = kernel.launch @memset_zero_2D(%out)
        : (tensor<?x?xf64>) -> tensor<?x?xf64>
    %result = kernel.launch @cublasDgemm(%a, %b, %zeroed, %beta, %alpha)
        : (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, f64, f64)
          -> tensor<?x?xf64>
    return %result : tensor<?x?xf64>
  }
}

// CHECK-LABEL: func.func @dynamic_beta
// CHECK: call @polygeist_cublas_dgemm
// CHECK-SAME: %arg4, {{.*}}, %arg3,
// CHECK-NOT: kernel.launch

// -----

// A finite constant beta and uniformly-zero destination can use beta=0 even
// though the semantic matcher emitted the ordinary parameterized GEMM symbol.
module {
  kernel.defn @memset_zero_2D(%out: tensor<?x?xf64>) -> tensor<?x?xf64> {
    kernel.yield %out : tensor<?x?xf64>
  }
  kernel.defn @cublasDgemm(
      %a: tensor<?x?xf64>, %b: tensor<?x?xf64>, %out: tensor<?x?xf64>,
      %beta: f64, %alpha: f64) -> tensor<?x?xf64> {
    kernel.yield %out : tensor<?x?xf64>
  }

  func.func @constant_beta(
      %a: tensor<?x?xf64>, %b: tensor<?x?xf64>, %out: tensor<?x?xf64>,
      %alpha: f64) -> tensor<?x?xf64> {
    %two = arith.constant 2.000000e+00 : f64
    %zeroed = kernel.launch @memset_zero_2D(%out)
        : (tensor<?x?xf64>) -> tensor<?x?xf64>
    %result = kernel.launch @cublasDgemm(%a, %b, %zeroed, %two, %alpha)
        : (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, f64, f64)
          -> tensor<?x?xf64>
    return %result : tensor<?x?xf64>
  }
}

// CHECK-LABEL: func.func @constant_beta
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: call @polygeist_cublas_dgemm
// CHECK-SAME: %[[ZERO]]
// CHECK-NOT: kernel.launch
