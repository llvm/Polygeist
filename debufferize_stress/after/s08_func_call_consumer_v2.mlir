module {
  func.func private @sink(memref<?xf64>)
  func.func @call_consumer(%arg0: index, %arg1: memref<?xf64>) {
    %cst = arith.constant 1.000000e+00 : f64
    %c0 = arith.constant 0 : index
    memref.store %cst, %arg1[%c0] : memref<?xf64>
    call @sink(%arg1) : (memref<?xf64>) -> ()
    return
  }
}

