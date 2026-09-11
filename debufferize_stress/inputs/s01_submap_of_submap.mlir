// INTENT: two-level submap chain (submap of submap). The pass's
// traceSubmapChainToRoot collects both, but the re-emission only uses the
// LEAF submap's map + operands. Intermediate level is dropped — expect
// either silently wrong IR or a verifier failure.

#map = affine_map<(d0) -> (d0)>
module {
  func.func @submap_of_submap(%n: index, %s1: index, %s2: index,
                              %x: memref<?xf64>, %y: memref<?xf64>) {
    // Outer submap (stride s1)
    %xa = polygeist.submap(%x, %s1, %n) {map = affine_map<(d0)[s0] -> (d0 * s0)>}
        : (memref<?xf64>, index, index) -> memref<?xf64>
    %ya = polygeist.submap(%y, %s1, %n) {map = affine_map<(d0)[s0] -> (d0 * s0)>}
        : (memref<?xf64>, index, index) -> memref<?xf64>
    // Inner submap on top (stride s2 on the already-strided view)
    %xb = polygeist.submap(%xa, %s2, %n) {map = affine_map<(d0)[s0] -> (d0 * s0)>}
        : (memref<?xf64>, index, index) -> memref<?xf64>
    %yb = polygeist.submap(%ya, %s2, %n) {map = affine_map<(d0)[s0] -> (d0 * s0)>}
        : (memref<?xf64>, index, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
        ins(%xb : memref<?xf64>) outs(%yb : memref<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %v = arith.addf %in, %out : f64
      linalg.yield %v : f64
    }
    return
  }
}
