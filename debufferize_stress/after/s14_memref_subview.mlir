module {
  func.func @subview_then_store(%arg0: memref<8x8xf64>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f64
    %subview = memref.subview %arg0[0, 0] [4, 4] [1, 1] : memref<8x8xf64> to memref<4x4xf64, strided<[8, 1]>>
    memref.store %cst, %subview[%c0, %c0] : memref<4x4xf64, strided<[8, 1]>>
    return
  }
}

