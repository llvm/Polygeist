module {
  func.func @two_blocks(%arg0: i1, %arg1: memref<8xf64>) {
    %cst = arith.constant 1.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    memref.store %cst, %arg1[%c0] : memref<8xf64>
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    memref.store %cst, %arg1[%c1] : memref<8xf64>
    return
  }
}

