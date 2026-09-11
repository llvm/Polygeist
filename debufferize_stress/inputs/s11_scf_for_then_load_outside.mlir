// INTENT: scf.for body modifies the memref via stores in every iteration,
// then *after* the loop the same memref is loaded. The loop must be
// rewritten with an iter_arg + yield; the post-loop load must extract from
// the loop's *result*, not from the original to_tensor.

module {
  func.func @for_then_load(%n: index, %x: memref<?xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v  = arith.constant 1.0 : f64
    scf.for %i = %c0 to %n step %c1 {
      memref.store %v, %x[%i] : memref<?xf64>
    }
    %out = memref.load %x[%c0] : memref<?xf64>
    return %out : f64
  }
}
