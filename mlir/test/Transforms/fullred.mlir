// RUN: mlir-opt %s -detect-reduction -mem2reg | FileCheck %s
// RUN: mlir-opt %s -detect-reduction -memref-dataflow-opt | FileCheck %s
// XFAIL: *
module  {
  func @kernel_2mm(%arg0: f64, %arg1: memref<800x900xf64>, %arg2: memref<800x1100xf64>, %arg3: memref<1100x900xf64>) {
    %0 = memref.alloca() : memref<1xf64>
    affine.for %arg4 = 0 to 800 {
      affine.for %arg5 = 0 to 900 {
        %1 = affine.load %arg1[%arg4, %arg5] : memref<800x900xf64>
        affine.store %1, %0[0] : memref<1xf64>
        affine.for %arg6 = 0 to 1100 {
          %3 = affine.load %0[0] : memref<1xf64>
          %4 = affine.load %arg2[%arg4, %arg6] : memref<800x1100xf64>
          %5 = mulf %arg0, %4 : f64
          %6 = affine.load %arg3[%arg6, %arg5] : memref<1100x900xf64>
          %7 = mulf %5, %6 : f64
          %8 = addf %3, %7 : f64
          affine.store %8, %0[0] : memref<1xf64>
        }
        %2 = affine.load %0[0] : memref<1xf64>
        affine.store %2, %arg1[%arg4, %arg5] : memref<800x900xf64>
      }
    }
    return
  }
}
