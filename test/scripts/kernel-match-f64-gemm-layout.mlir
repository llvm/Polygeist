// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | sed '/^\/\/ CHECK/d' | FileCheck %s

module {
  kernel.defn @cublasDgemm_simple(
      %a: tensor<?x?xf64>, %b: tensor<?x?xf64>, %c: tensor<?x?xf64>)
      -> tensor<?x?xf64> {
    kernel.yield %c : tensor<?x?xf64>
  }

  func.func @transposed_a(
      %a: tensor<?x?xf64>, %b: tensor<?x?xf64>, %c: tensor<?x?xf64>)
      -> tensor<?x?xf64> {
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d2, d0)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%c : tensor<?x?xf64>) {
    ^bb0(%av: f64, %bv: f64, %out: f64):
      %p = arith.mulf %av, %bv : f64
      %sum = arith.addf %out, %p : f64
      linalg.yield %sum : f64
    } -> tensor<?x?xf64>
    return %r : tensor<?x?xf64>
  }

  func.func @transposed_b(
      %a: tensor<?x?xf64>, %b: tensor<?x?xf64>, %c: tensor<?x?xf64>)
      -> tensor<?x?xf64> {
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%c : tensor<?x?xf64>) {
    ^bb0(%av: f64, %bv: f64, %out: f64):
      %p = arith.mulf %av, %bv : f64
      %sum = arith.addf %out, %p : f64
      linalg.yield %sum : f64
    } -> tensor<?x?xf64>
    return %r : tensor<?x?xf64>
  }
}

// CHECK-LABEL: func.func @transposed_a
// CHECK: kernel.launch @cublasDgemm_simple
// CHECK-SAME: polygeist.gemm_trans_a = true
// CHECK-SAME: polygeist.gemm_trans_b = false
// CHECK-LABEL: func.func @transposed_b
// CHECK: kernel.launch @cublasDgemm_simple
// CHECK-SAME: polygeist.gemm_trans_a = false
// CHECK-SAME: polygeist.gemm_trans_b = true
