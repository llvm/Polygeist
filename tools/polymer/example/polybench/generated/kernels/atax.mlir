module  {
  func @kernel_atax(%arg0: i32, %arg1: i32, %arg2: memref<1900x2100xf64>, %arg3: memref<2100xf64>, %arg4: memref<2100xf64>, %arg5: memref<1900xf64>) {
    %c0_i32 = constant 0 : i32
    %cst = constant 0.000000e+00 : f64
    %0 = index_cast %arg1 : i32 to index
    %1 = sitofp %c0_i32 : i32 to f64
    affine.for %arg6 = 0 to %0 {
      affine.store %1, %arg4[%arg6] : memref<2100xf64>
    }
    %2 = index_cast %arg0 : i32 to index
    affine.for %arg6 = 0 to %2 {
      affine.store %cst, %arg5[%arg6] : memref<1900xf64>
      %3 = affine.load %arg5[%arg6] : memref<1900xf64>
      affine.for %arg7 = 0 to %0 {
        %5 = affine.load %arg2[%arg6, %arg7] : memref<1900x2100xf64>
        %6 = affine.load %arg3[%arg7] : memref<2100xf64>
        %7 = mulf %5, %6 : f64
        %8 = addf %3, %7 : f64
        affine.store %8, %arg5[%arg6] : memref<1900xf64>
      }
      %4 = affine.load %arg5[%arg6] : memref<1900xf64>
      affine.for %arg7 = 0 to %0 {
        %5 = affine.load %arg4[%arg7] : memref<2100xf64>
        %6 = affine.load %arg2[%arg6, %arg7] : memref<1900x2100xf64>
        %7 = mulf %6, %4 : f64
        %8 = addf %5, %7 : f64
        affine.store %8, %arg4[%arg7] : memref<2100xf64>
      }
    }
    return
  }
}
