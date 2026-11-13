module {
  func.func @then_load_after_store(%arg0: i1, %arg1: memref<8xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 3.140000e+00 : f64
    %0 = bufferization.to_tensor %arg1 : memref<8xf64>
    %1:2 = scf.if %arg0 -> (f64, tensor<8xf64>) {
      %inserted = tensor.insert %cst into %0[%c0] : tensor<8xf64>
      %extracted = tensor.extract %inserted[%c0] : tensor<8xf64>
      scf.yield %extracted, %inserted : f64, tensor<8xf64>
    } else {
      %extracted = tensor.extract %0[%c0] : tensor<8xf64>
      scf.yield %extracted, %0 : f64, tensor<8xf64>
    }
    %2 = bufferization.to_memref %1#1 : memref<8xf64>
    memref.copy %2, %arg1 : memref<8xf64> to memref<8xf64>
    return %1#0 : f64
  }
}

