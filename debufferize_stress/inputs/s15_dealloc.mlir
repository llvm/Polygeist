// INTENT: an alloc is followed by a store and finally a dealloc. The dealloc
// is not in the supported-user set. Check whether the pass handles dealloc
// gracefully (it has no tensor analogue) or chokes.

module {
  func.func @with_dealloc(%n: index) {
    %a = memref.alloc(%n) : memref<?xf64>
    %v = arith.constant 1.0 : f64
    %i0 = arith.constant 0 : index
    memref.store %v, %a[%i0] : memref<?xf64>
    memref.dealloc %a : memref<?xf64>
    return
  }
}
