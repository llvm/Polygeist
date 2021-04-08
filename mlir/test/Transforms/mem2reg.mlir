// RUN: mlir-opt --mem2reg %s | FileCheck %s
// XFAIL: *
module  {
  func @main(%arg0: memref<40xf64>) {
    %0 = memref.alloca() : memref<1xf64>
    %pre = affine.load %0[0] : memref<1xf64>
    call @S2(%0) : (memref<1xf64>) -> ()
    %1 = affine.load %0[0] : memref<1xf64>
    affine.for %arg1 = 1 to 10 {
      %2 = affine.load %0[0] : memref<1xf64>
      affine.store %2, %arg0[%arg1] : memref<40xf64>
    }
    return
  }
  func private @S2(%arg0: memref<1xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %cst = constant 0.000000e+00 : f64
    memref.store %cst, %arg0[%c0] : memref<1xf64>
    return
  }
}
