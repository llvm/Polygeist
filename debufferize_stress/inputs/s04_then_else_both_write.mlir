// INTENT: scf.if where BOTH branches store to the same memref at different
// indices. Each branch's result must yield from its own modified tensor;
// the eventual if must have a single result tensor that downstream code uses.

module {
  func.func @then_else_both_write(%cond: i1, %x: memref<8xf64>) {
    %v1 = arith.constant 1.0 : f64
    %v2 = arith.constant 2.0 : f64
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    scf.if %cond {
      memref.store %v1, %x[%i0] : memref<8xf64>
    } else {
      memref.store %v2, %x[%i1] : memref<8xf64>
    }
    return
  }
}
