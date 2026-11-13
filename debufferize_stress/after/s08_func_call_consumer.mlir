module {
  func.func private @sink(memref<?xf64>)
  func.func @call_consumer(%arg0: index, %arg1: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f64
    %0 = bufferization.to_tensor %arg1 : memref<?xf64>
    %inserted = tensor.insert %cst into %0[%c0] : tensor<?xf64>
    %1 = bufferization.to_memref %inserted : memref<?xf64>
    memref.copy %1, %arg1 : memref<?xf64> to memref<?xf64>
    call @sink(%arg1) : (memref<?xf64>) -> ()
    return
  }
}

