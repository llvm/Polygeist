module  {
  func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<800x900xf64>, %arg7: memref<800x1100xf64>, %arg8: memref<1100x900xf64>, %arg9: memref<900x1200xf64>, %arg10: memref<800x1200xf64>) {
    %cst = constant 0.000000e+00 : f64
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    %2 = index_cast %arg2 : i32 to index
    affine.for %arg11 = 0 to %0 {
      affine.for %arg12 = 0 to %1 {
        affine.store %cst, %arg6[%arg11, %arg12] : memref<800x900xf64>
        %4 = affine.load %arg6[%arg11, %arg12] : memref<800x900xf64>
        affine.for %arg13 = 0 to %2 {
          %5 = affine.load %arg7[%arg11, %arg13] : memref<800x1100xf64>
          %6 = mulf %arg4, %5 : f64
          %7 = affine.load %arg8[%arg13, %arg12] : memref<1100x900xf64>
          %8 = mulf %6, %7 : f64
          %9 = addf %4, %8 : f64
          affine.store %9, %arg6[%arg11, %arg12] : memref<800x900xf64>
        }
      }
    }
    %3 = index_cast %arg3 : i32 to index
    affine.for %arg11 = 0 to %0 {
      affine.for %arg12 = 0 to %3 {
        %4 = affine.load %arg10[%arg11, %arg12] : memref<800x1200xf64>
        %5 = mulf %4, %arg5 : f64
        affine.store %5, %arg10[%arg11, %arg12] : memref<800x1200xf64>
        %6 = affine.load %arg10[%arg11, %arg12] : memref<800x1200xf64>
        affine.for %arg13 = 0 to %1 {
          %7 = affine.load %arg6[%arg11, %arg13] : memref<800x900xf64>
          %8 = affine.load %arg9[%arg13, %arg12] : memref<900x1200xf64>
          %9 = mulf %7, %8 : f64
          %10 = addf %6, %9 : f64
          affine.store %10, %arg10[%arg11, %arg12] : memref<800x1200xf64>
        }
      }
    }
    return
  }
}

