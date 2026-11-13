module {
  func.func @two_blocks(%arg0: i1, %arg1: memref<8xf64>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f64
    %0 = bufferization.to_tensor %arg1 : memref<8xf64>
    %inserted = tensor.insert %cst into %0[%c0] : tensor<8xf64>
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %inserted_0 = tensor.insert %cst into %inserted[%c1] : tensor<8xf64>
    %1 = bufferization.to_memref %inserted_0 : memref<8xf64>
    memref.copy %1, %arg1 : memref<8xf64> to memref<8xf64>
    return
  }
}

