// INTENT: linalg.generic on a memref nested inside scf.if (sgemm pattern).
// The threading must lift the generic's result through the if's branches.

#map = affine_map<(d0) -> (d0)>
module {
  func.func @linalg_inside_if(%cond: i1, %n: index, %y: memref<?xf64>) {
    %v = arith.constant 1.0 : f64
    scf.if %cond {
      %ys = polygeist.submap(%y, %n) {map = #map}
          : (memref<?xf64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]}
          outs(%ys : memref<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %v : f64
      }
    }
    return
  }
}
