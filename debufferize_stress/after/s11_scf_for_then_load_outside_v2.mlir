module {
  func.func @for_then_load(%arg0: index, %arg1: memref<?xf64>) -> f64 {
    %cst = arith.constant 1.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf64>
    %1 = scf.for %arg2 = %c0 to %arg0 step %c1 iter_args(%arg3 = %0) -> (tensor<?xf64>) {
      %inserted = tensor.insert %cst into %arg3[%arg2] : tensor<?xf64>
      scf.yield %inserted : tensor<?xf64>
    }
    %2 = bufferization.to_memref %1 : memref<?xf64>
    memref.copy %2, %arg1 : memref<?xf64> to memref<?xf64>
    %extracted = tensor.extract %1[%c0] : tensor<?xf64>
    return %extracted : f64
  }
}

