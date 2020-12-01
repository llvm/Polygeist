module  {
  func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<800x900xf64>, %arg6: memref<800x1000xf64>, %arg7: memref<1000x900xf64>, %arg8: memref<900x1100xf64>, %arg9: memref<900x1200xf64>, %arg10: memref<1200x1100xf64>, %arg11: memref<800x1100xf64>) {
    %cst = constant 0.000000e+00 : f64
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    %2 = index_cast %arg2 : i32 to index
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %1 {
        affine.store %cst, %arg5[%arg12, %arg13] : memref<800x900xf64>
        %5 = affine.load %arg5[%arg12, %arg13] : memref<800x900xf64>
        affine.for %arg14 = 0 to %2 {
          %6 = affine.load %arg6[%arg12, %arg14] : memref<800x1000xf64>
          %7 = affine.load %arg7[%arg14, %arg13] : memref<1000x900xf64>
          %8 = mulf %6, %7 : f64
          %9 = addf %5, %8 : f64
          affine.store %9, %arg5[%arg12, %arg13] : memref<800x900xf64>
        }
      }
    }
    %3 = index_cast %arg3 : i32 to index
    %4 = index_cast %arg4 : i32 to index
    affine.for %arg12 = 0 to %1 {
      affine.for %arg13 = 0 to %3 {
        affine.store %cst, %arg8[%arg12, %arg13] : memref<900x1100xf64>
        %5 = affine.load %arg8[%arg12, %arg13] : memref<900x1100xf64>
        affine.for %arg14 = 0 to %4 {
          %6 = affine.load %arg9[%arg12, %arg14] : memref<900x1200xf64>
          %7 = affine.load %arg10[%arg14, %arg13] : memref<1200x1100xf64>
          %8 = mulf %6, %7 : f64
          %9 = addf %5, %8 : f64
          affine.store %9, %arg8[%arg12, %arg13] : memref<900x1100xf64>
        }
      }
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %3 {
        affine.store %cst, %arg11[%arg12, %arg13] : memref<800x1100xf64>
        %5 = affine.load %arg11[%arg12, %arg13] : memref<800x1100xf64>
        affine.for %arg14 = 0 to %1 {
          %6 = affine.load %arg5[%arg12, %arg14] : memref<800x900xf64>
          %7 = affine.load %arg8[%arg14, %arg13] : memref<900x1100xf64>
          %8 = mulf %6, %7 : f64
          %9 = addf %5, %8 : f64
          affine.store %9, %arg11[%arg12, %arg13] : memref<800x1100xf64>
        }
      }
    }
    return
  }
}
