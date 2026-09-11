// INTENT: linalg.generic writes to a strided view, then a memref.load reads
// from the same root outside any region. Tests that the linalg's tensor
// result is correctly threaded to the subsequent load.

#map = affine_map<(d0) -> (d0)>
module {
  func.func @linalg_then_load(%n: index, %x: memref<?xf64>,
                              %y: memref<?xf64>) -> f64 {
    %xs = polygeist.submap(%x, %n) {map = #map}
        : (memref<?xf64>, index) -> memref<?xf64>
    %ys = polygeist.submap(%y, %n) {map = #map}
        : (memref<?xf64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
        ins(%xs : memref<?xf64>) outs(%ys : memref<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %v = arith.addf %in, %out : f64
      linalg.yield %v : f64
    }
    %c0 = arith.constant 0 : index
    %r = memref.load %y[%c0] : memref<?xf64>
    return %r : f64
  }
}
