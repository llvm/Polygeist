// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | sed '/^\/\/ CHECK/d' | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>
#map3 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @arbitrary_overwrite(%matrix: tensor<?x?xf64>,
                                 %vector: tensor<?xf64>,
                                 %scratch: tensor<?xf64>,
                                 %destination: tensor<?xf64>)
      -> tensor<?xf64> {
    %zero = arith.constant 0.000000e+00 : f64
    %init = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%scratch : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %zero : f64
    } -> tensor<?xf64>
    %product = linalg.generic {doc = "", indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%vector, %matrix : tensor<?xf64>, tensor<?x?xf64>) outs(%init : tensor<?xf64>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %mul = arith.mulf %in, %in_1 : f64
      %add = arith.addf %out, %mul : f64
      linalg.yield %add : f64
    } -> tensor<?xf64>
    %copy = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%product : tensor<?xf64>) outs(%destination : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<?xf64>
    return %copy : tensor<?xf64>
  }
}

// CHECK-LABEL: func.func @arbitrary_overwrite
// CHECK-NOT: linalg.generic
// CHECK: %[[R:.*]] = kernel.launch @cublasDgemv_T_zero(%matrix, %vector, %destination)
// CHECK: return %[[R]]
