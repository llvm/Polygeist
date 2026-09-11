module {
  func.func @sibling_ifs(%arg0: i1, %arg1: i1, %arg2: memref<8xf64>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %0 = bufferization.to_tensor %arg2 : memref<8xf64>
    %1 = scf.if %arg0 -> (tensor<8xf64>) {
      %inserted = tensor.insert %cst_0 into %0[%c0] : tensor<8xf64>
      scf.yield %inserted : tensor<8xf64>
    } else {
      scf.yield %0 : tensor<8xf64>
    }
    %2 = scf.if %arg1 -> (tensor<8xf64>) {
      %inserted = tensor.insert %cst into %1[%c1] : tensor<8xf64>
      scf.yield %inserted : tensor<8xf64>
    } else {
      scf.yield %1 : tensor<8xf64>
    }
    %3 = bufferization.to_memref %2 : memref<8xf64>
    memref.copy %3, %arg2 : memref<8xf64> to memref<8xf64>
    return
  }
}

