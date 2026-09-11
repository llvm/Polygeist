#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
module {
  func.func @reduction_then_load(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<?xf64>) -> f64 {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg2 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = tensor.empty() : tensor<f64>
    %inserted = tensor.insert %cst into %2[] : tensor<f64>
    %3 = polygeist.submap(%inserted, %arg0) {map = #map} : (tensor<f64>, index) -> tensor<?xf64>
    %4 = polygeist.submap(%1, %arg0) {map = #map1} : (tensor<?xf64>, index) -> tensor<?xf64>
    %5 = polygeist.submap(%0, %arg0) {map = #map1} : (tensor<?xf64>, index) -> tensor<?xf64>
    %6 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1], iterator_types = ["reduction"], library_call = ""} ins(%4, %5 : tensor<?xf64>, tensor<?xf64>) outs(%3 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %8 = arith.mulf %in, %in_0 : f64
      %9 = arith.addf %out, %8 : f64
      linalg.yield %9 : f64
    } -> tensor<?xf64>
    %7 = polygeist.submapInverse(%inserted, %6, %arg0) {map = #map} : (tensor<f64>, tensor<?xf64>, index) -> tensor<f64>
    %extracted = tensor.extract %7[] : tensor<f64>
    return %extracted : f64
  }
}

