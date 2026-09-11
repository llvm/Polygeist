// INTENT: affine.for has iter_args carrying a SCALAR plus stores to a
// memref in each iteration. The pass must handle iter_args on affine.for
// (propagateValueThroughRegion only adds iter_args for scf.for and yields
// for scf.if — there's NO affine.for branch).

module {
  func.func @affine_for_with_iter(%n: index, %x: memref<?xf64>) -> f64 {
    %cst = arith.constant 0.0 : f64
    %v   = arith.constant 1.0 : f64
    %s = affine.for %i = 0 to %n iter_args(%acc = %cst) -> f64 {
      memref.store %v, %x[%i] : memref<?xf64>
      %a = arith.addf %acc, %v : f64
      affine.yield %a : f64
    }
    return %s : f64
  }
}
