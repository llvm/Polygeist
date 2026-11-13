// INTENT: scf.if without an else region. The store happens conditionally;
// the rebuild must synthesize an else branch that yields the entry tensor.
// In propagateValueThroughRegion the `hadElse` path (line ~593) handles this,
// but the if-finalization codepath in the main loop assumes both branches
// were entered.

#map = affine_map<(d0) -> (d0)>
module {
  func.func @if_no_else(%cond: i1, %n: index, %x: memref<?xf64>) {
    %cst = arith.constant 1.0 : f64
    %c0 = arith.constant 0 : index
    scf.if %cond {
      memref.store %cst, %x[%c0] : memref<?xf64>
    }
    return
  }
}
