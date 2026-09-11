// INTENT: linalg.generic uses the SAME memref both as an input and as an
// output (read-then-write). Two submap chains both terminate at %x. The
// pass processes input and output operands separately, both pulling from
// `currentTensor`; the input submap and output submap will both reference
// the same SSA tensor pre-update. Verify the resulting IR is correct
// (the output should still update via submapInverse, and the input read
// should observe pre-update values per linalg semantics).

#map = affine_map<(d0) -> (d0)>
module {
  func.func @in_eq_out(%n: index, %x: memref<?xf64>) {
    %xs = polygeist.submap(%x, %n) {map = #map}
        : (memref<?xf64>, index) -> memref<?xf64>
    %xo = polygeist.submap(%x, %n) {map = #map}
        : (memref<?xf64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
        ins(%xs : memref<?xf64>) outs(%xo : memref<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %v = arith.mulf %in, %in : f64
      linalg.yield %v : f64
    }
    return
  }
}
