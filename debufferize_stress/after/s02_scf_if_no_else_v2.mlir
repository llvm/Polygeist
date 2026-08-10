module {
  func.func @if_no_else(%arg0: i1, %arg1: index, %arg2: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f64
    %0 = bufferization.to_tensor %arg2 : memref<?xf64>
    %1 = scf.if %arg0 -> (tensor<?xf64>) {
      %inserted = tensor.insert %cst into %0[%c0] : tensor<?xf64>
      scf.yield %inserted : tensor<?xf64>
    } else {
      scf.yield %0 : tensor<?xf64>
    }
    %2 = bufferization.to_memref %1 : memref<?xf64>
    memref.copy %2, %arg2 : memref<?xf64> to memref<?xf64>
    return
  }
}

