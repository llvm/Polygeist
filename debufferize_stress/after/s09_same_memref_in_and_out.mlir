#map = affine_map<(d0) -> (d0)>
module {
  func.func @in_eq_out(%arg0: index, %arg1: memref<?xf64>) {
    %0 = bufferization.to_tensor %arg1 : memref<?xf64>
    %1 = polygeist.submap(%0, %arg0) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %2 = polygeist.submap(%0, %arg0) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %3 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%1 : tensor<?xf64>) outs(%2 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %6 = arith.mulf %in, %in : f64
      linalg.yield %6 : f64
    } -> tensor<?xf64>
    %4 = polygeist.submapInverse(%0, %3, %arg0) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %5 = bufferization.to_memref %4 : memref<?xf64>
    memref.copy %5, %arg1 : memref<?xf64> to memref<?xf64>
    return
  }
}

