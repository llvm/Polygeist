// XFAIL: *
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --only-kernel=cutensornetContraction2_f64 | sed '/^[[:space:]]*\/\/ CHECK/d' | FileCheck %s

// Known safety gap: an adjacent zero-fill of an unrelated tensor must not
// establish overwrite (beta=0) semantics for a contraction that accumulates
// into another tensor.  Keep this focused test XFAIL until the rewrite proves
// the initializer-to-destination dataflow/alias chain.

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

module {
  // CHECK-LABEL: func.func @unrelated_zero_is_not_an_initializer
  // CHECK-NOT: kernel.launch @cutensornetContraction2_f64
  func.func @unrelated_zero_is_not_an_initializer(%a: tensor<?x?x?xf64>,
                                                   %b: tensor<?x?x?xf64>,
                                                   %decoy: tensor<?x?x?xf64>,
                                                   %target: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %zero = arith.constant 0.000000e+00 : f64
    %cleared_decoy = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%decoy : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %zero : f64
    } -> tensor<?x?x?xf64>
    %accumulated = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%a, %b : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%target : tensor<?x?x?xf64>) {
    ^bb0(%x: f64, %y: f64, %out: f64):
      %product = arith.mulf %x, %y : f64
      %sum = arith.addf %out, %product : f64
      linalg.yield %sum : f64
    } -> tensor<?x?x?xf64>
    return %accumulated : tensor<?x?x?xf64>
  }
}
