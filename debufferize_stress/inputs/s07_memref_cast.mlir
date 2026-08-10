// INTENT: a memref.cast user — pass's supported-user set doesn't include
// memref.cast. collectMemoryOpsRecursively will skip it, so the store
// reached via the cast is not in the user list at all → not rewritten.
// Expect: store untouched, function still memref-typed, no debufferization.

module {
  func.func @cast_then_store(%x: memref<8xf64>) {
    %xc = memref.cast %x : memref<8xf64> to memref<?xf64>
    %v = arith.constant 1.0 : f64
    %i0 = arith.constant 0 : index
    memref.store %v, %xc[%i0] : memref<?xf64>
    return
  }
}
