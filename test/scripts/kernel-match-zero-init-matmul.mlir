// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | sed '/^\/\/ CHECK/d' | FileCheck %s

module {
  func.func @arbitrary_dense_product(%a: tensor<?x?xf64>, %b: tensor<?x?xf64>, %c: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %zero = arith.constant 0.000000e+00 : f64
    %init = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        outs(%c : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %zero : f64
    } -> tensor<?x?xf64>
    %product = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<?x?xf64>, tensor<?x?xf64>)
        outs(%init : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %mul = arith.mulf %in, %in_8 : f64
      %add = arith.addf %out, %mul : f64
      linalg.yield %add : f64
    } -> tensor<?x?xf64>
    return %product : tensor<?x?xf64>
  }
}

// CHECK-LABEL: func.func @arbitrary_dense_product
// CHECK: %[[INIT:.*]] = kernel.launch @memset_zero_2D(%c)
// CHECK: %[[PRODUCT:.*]] = kernel.launch @cublasDgemm_simple(%a, %b, %c)
// CHECK: return %[[PRODUCT]]
// CHECK-NOT: cutensornetContraction2
