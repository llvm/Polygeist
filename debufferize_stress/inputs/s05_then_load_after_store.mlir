// INTENT: scf.if writes to %x then reads from it within the same branch.
// The load inside the THEN branch must extract from the just-modified
// tensor — i.e. the load should see the *insert's* result, not the
// pre-if entry tensor. Verifies intra-region threading.

module {
  func.func @then_load_after_store(%cond: i1, %x: memref<8xf64>) -> f64 {
    %v = arith.constant 3.14 : f64
    %i0 = arith.constant 0 : index
    %r = scf.if %cond -> f64 {
      memref.store %v, %x[%i0] : memref<8xf64>
      %l = memref.load %x[%i0] : memref<8xf64>
      scf.yield %l : f64
    } else {
      %l2 = memref.load %x[%i0] : memref<8xf64>
      scf.yield %l2 : f64
    }
    return %r : f64
  }
}
