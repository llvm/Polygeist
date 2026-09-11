// INTENT: memref.subview between alloca and store. memref.subview IS listed
// in areAllUsersSupportedForDebufferization but is NOT in
// collectMemoryOpsRecursively's recursion set (only polygeist.submap is).
// Expect: the store via the subview is NOT collected, so the alloca is
// effectively treated as having no users → debuf is a no-op for it.

module {
  func.func @subview_then_store(%x: memref<8x8xf64>) {
    %sv = memref.subview %x[0, 0] [4, 4] [1, 1] : memref<8x8xf64> to memref<4x4xf64, strided<[8, 1]>>
    %v = arith.constant 1.0 : f64
    %i0 = arith.constant 0 : index
    memref.store %v, %sv[%i0, %i0] : memref<4x4xf64, strided<[8, 1]>>
    return
  }
}
