#map = affine_map<(d0) -> (d0)>
module {
  func.func @linalg_then_load(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<?xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = polygeist.submap(%1, %arg0) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %3 = polygeist.submap(%0, %arg0) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %4 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%2 : tensor<?xf64>) outs(%3 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %7 = arith.addf %in, %out : f64
      linalg.yield %7 : f64
    } -> tensor<?xf64>
    %5 = polygeist.submapInverse(%0, %4, %arg0) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %6 = bufferization.to_memref %5 : memref<?xf64>
    memref.copy %6, %arg2 : memref<?xf64> to memref<?xf64>
    %extracted = tensor.extract %5[%c0] : tensor<?xf64>
    return %extracted : f64
  }
}

