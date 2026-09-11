// INTENT: scf.while contains memref stores. propagateValueThroughRegion only
// has codepaths for scf.if and scf.for — there's no scf.while branch. Expect
// the inner store to be rewritten but the tensor SSA threading to fail at
// finalization (the while's region won't get an iter_arg added).

module {
  func.func @scf_while_store(%n: index, %x: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v = arith.constant 1.0 : f64
    %final = scf.while (%i = %c0) : (index) -> index {
      %cond = arith.cmpi slt, %i, %n : index
      scf.condition(%cond) %i : index
    } do {
    ^bb0(%i: index):
      memref.store %v, %x[%i] : memref<?xf64>
      %next = arith.addi %i, %c1 : index
      scf.yield %next : index
    }
    return
  }
}
