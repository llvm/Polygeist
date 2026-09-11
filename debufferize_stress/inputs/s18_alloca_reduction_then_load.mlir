// INTENT: mirrors the post-`--remove-iter-args` shape — a 0-D scalar alloca
// used as the reduction destination inside a loop, then loaded after the
// loop. This is exactly what we just made the BLAS reductions emit.
// Verify the debufferizer handles the 0-D alloca-after-loop pattern end-to-end.

#map = affine_map<(d0) -> (d0)>
#m0  = affine_map<(d0) -> ()>
module {
  func.func @reduction_then_load(%n: index, %x: memref<?xf64>,
                                 %y: memref<?xf64>) -> f64 {
    %cst = arith.constant 0.0 : f64
    %slot = memref.alloca() : memref<f64>
    affine.store %cst, %slot[] : memref<f64>
    %xs = polygeist.submap(%x, %n) {map = #map}
        : (memref<?xf64>, index) -> memref<?xf64>
    %ys = polygeist.submap(%y, %n) {map = #map}
        : (memref<?xf64>, index) -> memref<?xf64>
    %ss = polygeist.submap(%slot, %n) {map = #m0}
        : (memref<f64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map, #map, #map],
                    iterator_types = ["reduction"]}
        ins(%xs, %ys : memref<?xf64>, memref<?xf64>) outs(%ss : memref<?xf64>) {
    ^bb0(%a: f64, %b: f64, %acc: f64):
      %p = arith.mulf %a, %b : f64
      %v = arith.addf %acc, %p : f64
      linalg.yield %v : f64
    }
    %r = affine.load %slot[] : memref<f64>
    return %r : f64
  }
}
