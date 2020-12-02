#map = affine_map<(d0) -> (d0)>
module  {
  func @kernel_symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1000xf64>, %arg6: memref<1000x1200xf64>) {
    %c0 = constant 0 : index
    %c0_i32 = constant 0 : i32
    %0 = alloca() : memref<1xf64>
    %1 = index_cast %arg0 : i32 to index
    %2 = index_cast %arg1 : i32 to index
    %3 = sitofp %c0_i32 : i32 to f64
    affine.store %3, %0[%c0] : memref<1xf64>
    %4 = affine.load %0[%c0] : memref<1xf64>
    %5 = affine.load %0[%c0] : memref<1xf64>
    %6 = mulf %arg2, %5 : f64
    affine.for %arg7 = 0 to %1 {
      %7 = affine.load %arg5[%arg7, %arg7] : memref<1000x1000xf64>
      affine.for %arg8 = 0 to %2 {
        %8 = affine.load %arg6[%arg7, %arg8] : memref<1000x1200xf64>
        %9 = mulf %arg2, %8 : f64
        affine.for %arg9 = 0 to #map(%arg7) {
          %17 = affine.load %arg5[%arg7, %arg9] : memref<1000x1000xf64>
          %18 = mulf %9, %17 : f64
          %19 = affine.load %arg4[%arg9, %arg8] : memref<1000x1200xf64>
          %20 = addf %19, %18 : f64
          affine.store %20, %arg4[%arg9, %arg8] : memref<1000x1200xf64>
          %21 = affine.load %arg6[%arg9, %arg8] : memref<1000x1200xf64>
          %22 = affine.load %arg5[%arg7, %arg9] : memref<1000x1000xf64>
          %23 = mulf %21, %22 : f64
          %24 = addf %4, %23 : f64
          affine.store %24, %0[%c0] : memref<1xf64>
        }
        %10 = affine.load %arg4[%arg7, %arg8] : memref<1000x1200xf64>
        %11 = mulf %arg3, %10 : f64
        %12 = affine.load %arg6[%arg7, %arg8] : memref<1000x1200xf64>
        %13 = mulf %arg2, %12 : f64
        %14 = mulf %13, %7 : f64
        %15 = addf %11, %14 : f64
        %16 = addf %15, %6 : f64
        affine.store %16, %arg4[%arg7, %arg8] : memref<1000x1200xf64>
      }
    }
    return
  }
}

