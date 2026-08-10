// INTENT: memref passed to a func.call. Call isn't in the supported-user
// set; the call may modify the memref in a way the pass can't track.
// Expect: pass either ignores the call (treats memref as not modified by it)
// or fails to debufferize. Test reveals which.

module {
  func.func private @sink(memref<?xf64>)
  func.func @call_consumer(%n: index, %x: memref<?xf64>) {
    %v = arith.constant 1.0 : f64
    %i0 = arith.constant 0 : index
    memref.store %v, %x[%i0] : memref<?xf64>
    func.call @sink(%x) : (memref<?xf64>) -> ()
    return
  }
}
