#map = affine_map<(d0) -> (d0)>
module {
  func.func @linalg_inside_if(%arg0: i1, %arg1: index, %arg2: memref<?xf64>) {
    %cst = arith.constant 1.000000e+00 : f64
    %0 = bufferization.to_tensor %arg2 : memref<?xf64>
    %1 = scf.if %arg0 -> (tensor<?xf64>) {
      %3 = polygeist.submap(%0, %arg1) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
      %4 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%3 : tensor<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      } -> tensor<?xf64>
      %5 = polygeist.submapInverse(%0, %4, %arg1) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
      scf.yield %5 : tensor<?xf64>
    } else {
      scf.yield %0 : tensor<?xf64>
    }
    %2 = bufferization.to_memref %1 : memref<?xf64>
    memref.copy %2, %arg2 : memref<?xf64> to memref<?xf64>
    return
  }
}

