#map = affine_map<(d0) -> (d0)>
module  {
  func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<1400x1200xf64>, %arg4: memref<1200x1200xf64>, %arg5: memref<1200xf64>) {
    %cst = constant 0.000000e+00 : f64
    %cst_0 = constant 1.000000e+00 : f64
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg6 = 0 to %0 {
      affine.store %cst, %arg5[%arg6] : memref<1200xf64>
      %3 = affine.load %arg5[%arg6] : memref<1200xf64>
      affine.for %arg7 = 0 to %1 {
        %6 = affine.load %arg3[%arg7, %arg6] : memref<1400x1200xf64>
        %7 = addf %3, %6 : f64
        affine.store %7, %arg5[%arg6] : memref<1200xf64>
      }
      %4 = affine.load %arg5[%arg6] : memref<1200xf64>
      %5 = divf %4, %arg2 : f64
      affine.store %5, %arg5[%arg6] : memref<1200xf64>
    }
    affine.for %arg6 = 0 to %1 {
      affine.for %arg7 = 0 to %0 {
        %3 = affine.load %arg5[%arg7] : memref<1200xf64>
        %4 = affine.load %arg3[%arg6, %arg7] : memref<1400x1200xf64>
        %5 = subf %4, %3 : f64
        affine.store %5, %arg3[%arg6, %arg7] : memref<1400x1200xf64>
      }
    }
    %2 = subf %arg2, %cst_0 : f64
    affine.for %arg6 = 0 to %0 {
      affine.for %arg7 = #map(%arg6) to %0 {
        affine.store %cst, %arg4[%arg6, %arg7] : memref<1200x1200xf64>
        %3 = affine.load %arg4[%arg6, %arg7] : memref<1200x1200xf64>
        affine.for %arg8 = 0 to %1 {
          %7 = affine.load %arg3[%arg8, %arg6] : memref<1400x1200xf64>
          %8 = affine.load %arg3[%arg8, %arg7] : memref<1400x1200xf64>
          %9 = mulf %7, %8 : f64
          %10 = addf %3, %9 : f64
          affine.store %10, %arg4[%arg6, %arg7] : memref<1200x1200xf64>
        }
        %4 = affine.load %arg4[%arg6, %arg7] : memref<1200x1200xf64>
        %5 = divf %4, %2 : f64
        affine.store %5, %arg4[%arg6, %arg7] : memref<1200x1200xf64>
        %6 = affine.load %arg4[%arg6, %arg7] : memref<1200x1200xf64>
        affine.store %6, %arg4[%arg7, %arg6] : memref<1200x1200xf64>
      }
    }
    return
  }
}
