#map = affine_map<(d0)[s0] -> (d0 * s0)>
#map1 = affine_map<(d0) -> (d0)>
module {
  func.func @submap_of_submap(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<?xf64>, %arg4: memref<?xf64>) {
    %0 = bufferization.to_tensor %arg4 : memref<?xf64>
    %1 = bufferization.to_tensor %arg3 : memref<?xf64>
    %2 = polygeist.submap(%1, %arg1, %arg0) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?xf64>
    %3 = polygeist.submap(%2, %arg2, %arg0) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?xf64>
    %4 = polygeist.submap(%0, %arg1, %arg0) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?xf64>
    %5 = polygeist.submap(%4, %arg2, %arg0) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?xf64>
    %6 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%3 : tensor<?xf64>) outs(%5 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %11 = arith.addf %in, %out : f64
      linalg.yield %11 : f64
    } -> tensor<?xf64>
    %7 = polygeist.submap(%0, %arg1, %arg0) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?xf64>
    %8 = polygeist.submapInverse(%7, %6, %arg2, %arg0) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index, index) -> tensor<?xf64>
    %9 = polygeist.submapInverse(%0, %8, %arg1, %arg0) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index, index) -> tensor<?xf64>
    %10 = bufferization.to_memref %9 : memref<?xf64>
    memref.copy %10, %arg4 : memref<?xf64> to memref<?xf64>
    return
  }
}

