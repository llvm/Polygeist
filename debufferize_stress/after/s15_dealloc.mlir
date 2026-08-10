module {
  func.func @with_dealloc(%arg0: index) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc(%arg0) : memref<?xf64>
    %0 = bufferization.to_tensor %alloc : memref<?xf64>
    %inserted = tensor.insert %cst into %0[%c0] : tensor<?xf64>
    %1 = bufferization.to_memref %inserted : memref<?xf64>
    memref.copy %1, %alloc : memref<?xf64> to memref<?xf64>
    memref.dealloc %alloc : memref<?xf64>
    return
  }
}

