module {
  func.func @cast_then_store(%arg0: memref<8xf64>) {
    %cst = arith.constant 1.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg0 : memref<8xf64>
    %inserted = tensor.insert %cst into %0[%c0] : tensor<8xf64>
    %1 = bufferization.to_memref %inserted : memref<8xf64>
    memref.copy %1, %arg0 : memref<8xf64> to memref<8xf64>
    return
  }
}

