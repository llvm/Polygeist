// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | sed '/^\/\/ CHECK/d' | FileCheck %s

module {
  func.func @rank_zero_dot(%x: tensor<?xf32>, %y: tensor<?xf32>,
                           %out: tensor<f32>) -> tensor<f32> {
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]}
        ins(%x, %y : tensor<?xf32>, tensor<?xf32>)
        outs(%out : tensor<f32>) {
    ^bb0(%xv: f32, %yv: f32, %out_value: f32):
      %product = arith.mulf %xv, %yv : f32
      %sum = arith.addf %out_value, %product : f32
      linalg.yield %sum : f32
    } -> tensor<f32>
    return %result : tensor<f32>
  }

  func.func @zero_init_batched_sgemm(
      %a: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>,
      %out: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %zero = arith.constant 0.0 : f32
    %initialized = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]}
        outs(%out : tensor<?x?x?xf32>) {
    ^bb0(%out_value: f32):
      linalg.yield %zero : f32
    } -> tensor<?x?x?xf32>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%initialized : tensor<?x?x?xf32>) {
    ^bb0(%av: f32, %bv: f32, %out_value: f32):
      %product = arith.mulf %av, %bv : f32
      %sum = arith.addf %out_value, %product : f32
      linalg.yield %sum : f32
    } -> tensor<?x?x?xf32>
    return %result : tensor<?x?x?xf32>
  }

  // The current memref ABI intentionally represents a scalar destination as
  // a rank-one aliasing view.  A true rank-zero memref needs separate ABI
  // support and must not be accepted through the tensor rank-zero path.
  func.func @rank_zero_memref_dot_not_abi_compatible(
      %x: memref<?xf32>, %y: memref<?xf32>, %out: memref<f32>) {
    linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]}
        ins(%x, %y : memref<?xf32>, memref<?xf32>)
        outs(%out : memref<f32>) {
    ^bb0(%xv: f32, %yv: f32, %out_value: f32):
      %product = arith.mulf %xv, %yv : f32
      %sum = arith.addf %out_value, %product : f32
      linalg.yield %sum : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @rank_zero_dot
// CHECK: kernel.launch @cublasSdot
// CHECK-NOT: linalg.generic
// CHECK-LABEL: func.func @zero_init_batched_sgemm
// CHECK: kernel.launch @cublasSgemm_strided_batched_nn_zero
// CHECK-NOT: linalg.generic
// CHECK-LABEL: func.func @rank_zero_memref_dot_not_abi_compatible
// CHECK: linalg.generic
