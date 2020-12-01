module  {
  func @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<1000x1100xf64>, %arg6: memref<1000x1200xf64>, %arg7: memref<1200x1100xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    %2 = index_cast %arg2 : i32 to index
    affine.for %arg8 = 0 to %0 {
      affine.for %arg9 = 0 to %1 {
        %3 = affine.load %arg5[%arg8, %arg9] : memref<1000x1100xf64>
        %4 = mulf %3, %arg4 : f64
        affine.store %4, %arg5[%arg8, %arg9] : memref<1000x1100xf64>
      }
      affine.for %arg9 = 0 to %2 {
        %3 = affine.load %arg6[%arg8, %arg9] : memref<1000x1200xf64>
        %4 = mulf %arg3, %3 : f64
        affine.for %arg10 = 0 to %1 {
          %5 = affine.load %arg7[%arg9, %arg10] : memref<1200x1100xf64>
          %6 = mulf %4, %5 : f64
          %7 = affine.load %arg5[%arg8, %arg10] : memref<1000x1100xf64>
          %8 = addf %7, %6 : f64
          affine.store %8, %arg5[%arg8, %arg10] : memref<1000x1100xf64>
        }
      }
    }
    return
  }
}
