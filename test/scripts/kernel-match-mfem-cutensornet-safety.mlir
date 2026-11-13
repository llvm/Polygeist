// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --only-kernel=cutensornetContraction2_f64 | sed '/^[[:space:]]*\/\/ CHECK/d' | FileCheck %s

// Regression coverage for the generic FP64 contraction route used by the
// normalized MFEM fixtures.  Recognition must depend on tensor dataflow,
// scalar reduction semantics, and affine-map legality, never symbol names.

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d1)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 8 + d2)>

module {
  // CHECK-LABEL: func.func @first_arbitrary_name
  // CHECK: kernel.launch @cutensornetContraction2_f64
  func.func @first_arbitrary_name(%a: tensor<?x?x?xf64>,
                                  %b: tensor<?x?x?xf64>,
                                  %c: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %zero = arith.constant 0.000000e+00 : f64
    %init = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%c : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %zero : f64
    } -> tensor<?x?x?xf64>
    %contracted = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%a, %b : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%init : tensor<?x?x?xf64>) {
    ^bb0(%x: f64, %y: f64, %out: f64):
      %product = arith.mulf %x, %y : f64
      %sum = arith.addf %out, %product : f64
      linalg.yield %sum : f64
    } -> tensor<?x?x?xf64>
    return %contracted : tensor<?x?x?xf64>
  }

  // Same semantics with every source-visible symbol, argument, and operation
  // SSA name changed. The parser's conventional output block name remains
  // `%out`; both positives must select the same external-library opcode.
  // CHECK-LABEL: func.func @completely_renamed
  // CHECK: kernel.launch @cutensornetContraction2_f64
  func.func @completely_renamed(%left: tensor<?x?x?xf64>,
                                %right: tensor<?x?x?xf64>,
                                %destination: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %nothing = arith.constant 0.000000e+00 : f64
    %cleared = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%destination : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %nothing : f64
    } -> tensor<?x?x?xf64>
    %answer = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%left, %right : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%cleared : tensor<?x?x?xf64>) {
    ^bb0(%one: f64, %two: f64, %out: f64):
      %times = arith.mulf %two, %one : f64
      %plus = arith.addf %times, %out : f64
      linalg.yield %plus : f64
    } -> tensor<?x?x?xf64>
    return %answer : tensor<?x?x?xf64>
  }

  // A nonzero fill is not an overwrite-mode contraction initializer.
  // CHECK-LABEL: func.func @nonzero_accumulator
  // CHECK-NOT: kernel.launch @cutensornetContraction2_f64
  func.func @nonzero_accumulator(%a: tensor<?x?x?xf64>,
                                 %b: tensor<?x?x?xf64>,
                                 %c: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %one = arith.constant 1.000000e+00 : f64
    %init = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%c : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %one : f64
    } -> tensor<?x?x?xf64>
    %accumulated = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%a, %b : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%init : tensor<?x?x?xf64>) {
    ^bb0(%x: f64, %y: f64, %out: f64):
      %product = arith.mulf %x, %y : f64
      %sum = arith.addf %out, %product : f64
      linalg.yield %sum : f64
    } -> tensor<?x?x?xf64>
    return %accumulated : tensor<?x?x?xf64>
  }

  // A logical reduction mode in the output is legal when its physical view
  // has zero stride for that mode, as in MFEM scratch-sliced staging.
  // CHECK-LABEL: func.func @proven_physical_broadcast
  // CHECK: kernel.launch @cutensornetContraction2_f64
  func.func @proven_physical_broadcast(%a: tensor<?x?x?xf64>,
                                       %b: tensor<?x?x?xf64>,
                                       %storage: tensor<?x?x?xf64>) -> tensor<?x?x?x?xf64> {
    %zero = arith.constant 0.000000e+00 : f64
    %init = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%storage : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %zero : f64
    } -> tensor<?x?x?xf64>
    %view = polygeist.submap(%init) {map = #map6} : (tensor<?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %contracted = linalg.generic {indexing_maps = [#map1, #map2, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%a, %b : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%view : tensor<?x?x?x?xf64>) {
    ^bb0(%x: f64, %y: f64, %out: f64):
      %product = arith.mulf %x, %y : f64
      %sum = arith.addf %out, %product : f64
      linalg.yield %sum : f64
    } -> tensor<?x?x?x?xf64>
    return %contracted : tensor<?x?x?x?xf64>
  }

  // The reduction mode cannot be a physical output mode unless a zero-stride
  // submap proves that it is only a logical broadcast dimension.
  // CHECK-LABEL: func.func @illegal_reduction_output_mode
  // CHECK-NOT: kernel.launch @cutensornetContraction2_f64
  func.func @illegal_reduction_output_mode(%a: tensor<?x?x?xf64>,
                                           %b: tensor<?x?x?xf64>,
                                           %c: tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64> {
    %zero = arith.constant 0.000000e+00 : f64
    %init = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%c : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %zero : f64
    } -> tensor<?x?x?x?xf64>
    %bad = linalg.generic {indexing_maps = [#map1, #map2, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%a, %b : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%init : tensor<?x?x?x?xf64>) {
    ^bb0(%x: f64, %y: f64, %out: f64):
      %product = arith.mulf %x, %y : f64
      %sum = arith.addf %out, %product : f64
      linalg.yield %sum : f64
    } -> tensor<?x?x?x?xf64>
    return %bad : tensor<?x?x?x?xf64>
  }

  // Duplicate output modes are not a legal cuTensorNet descriptor.
  // CHECK-LABEL: func.func @duplicate_output_mode
  // CHECK-NOT: kernel.launch @cutensornetContraction2_f64
  func.func @duplicate_output_mode(%a: tensor<?x?x?xf64>,
                                   %b: tensor<?x?x?xf64>,
                                   %c: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %zero = arith.constant 0.000000e+00 : f64
    %init = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%c : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %zero : f64
    } -> tensor<?x?x?xf64>
    %bad = linalg.generic {indexing_maps = [#map1, #map2, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%a, %b : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%init : tensor<?x?x?xf64>) {
    ^bb0(%x: f64, %y: f64, %out: f64):
      %product = arith.mulf %x, %y : f64
      %sum = arith.addf %out, %product : f64
      linalg.yield %sum : f64
    } -> tensor<?x?x?xf64>
    return %bad : tensor<?x?x?xf64>
  }
}
