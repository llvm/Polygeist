module {
  func.func @scf_while_store(%arg0: index, %arg1: memref<?xf64>) {
    %cst = arith.constant 1.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf64>
    %1:2 = scf.while (%arg2 = %c0, %arg3 = %0) : (index, tensor<?xf64>) -> (index, tensor<?xf64>) {
      %3 = arith.cmpi slt, %arg2, %arg0 : index
      scf.condition(%3) %arg2, %arg3 : index, tensor<?xf64>
    } do {
    ^bb0(%arg2: index, %arg3: tensor<?xf64>):
      %inserted = tensor.insert %cst into %arg3[%arg2] : tensor<?xf64>
      %3 = arith.addi %arg2, %c1 : index
      scf.yield %3, %inserted : index, tensor<?xf64>
    }
    %2 = bufferization.to_memref %1#1 : memref<?xf64>
    memref.copy %2, %arg1 : memref<?xf64> to memref<?xf64>
    return
  }
}

