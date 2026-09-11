module {
  func.func @affine_for_with_iter(%arg0: index, %arg1: memref<?xf64>) -> f64 {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg1 : memref<?xf64>
    %1:2 = affine.for %arg2 = 0 to %arg0 iter_args(%arg3 = %cst_0, %arg4 = %0) -> (f64, tensor<?xf64>) {
      %inserted = tensor.insert %cst into %arg4[%arg2] : tensor<?xf64>
      %3 = arith.addf %arg3, %cst : f64
      affine.yield %3, %inserted : f64, tensor<?xf64>
    }
    %2 = bufferization.to_memref %1#1 : memref<?xf64>
    memref.copy %2, %arg1 : memref<?xf64> to memref<?xf64>
    return %1#0 : f64
  }
}

