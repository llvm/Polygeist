// INTENT: function body has TWO basic blocks (cf.br between them). The pass
// uses region->front() (e.g. line 480), assuming exactly one block. With two
// blocks the second block's stores are technically still in the same region
// but may not be properly handled.

module {
  func.func @two_blocks(%cond: i1, %x: memref<8xf64>) {
    %v = arith.constant 1.0 : f64
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    memref.store %v, %x[%i0] : memref<8xf64>
    cf.br ^bb1
  ^bb1:
    memref.store %v, %x[%i1] : memref<8xf64>
    return
  }
}
