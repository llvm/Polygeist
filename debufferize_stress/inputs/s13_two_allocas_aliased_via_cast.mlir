// INTENT: two allocas of different static shape are both viewed via cast
// onto the same dynamic shape, then a store happens through the cast view.
// The per-root iteration treats each alloca independently — the cast user
// breaks the assumption that all uses of an alloca are typed identically.

module {
  func.func @aliased_allocas() {
    %a = memref.alloca() : memref<4xf64>
    %b = memref.alloca() : memref<4xf64>
    %ac = memref.cast %a : memref<4xf64> to memref<?xf64>
    %bc = memref.cast %b : memref<4xf64> to memref<?xf64>
    %v = arith.constant 1.0 : f64
    %i0 = arith.constant 0 : index
    memref.store %v, %ac[%i0] : memref<?xf64>
    memref.store %v, %bc[%i0] : memref<?xf64>
    return
  }
}
