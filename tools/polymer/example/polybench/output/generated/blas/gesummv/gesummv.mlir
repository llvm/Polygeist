#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<()[s0] -> (s0)>


module {
  func @kernel_gesummv(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<1300x1300xf64>, %arg4: memref<1300x1300xf64>, %arg5: memref<1300xf64>, %arg6: memref<1300xf64>, %arg7: memref<1300xf64>) {
    %cst = constant 0.000000e+00 : f64
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg8 = 0 to %0 {
      affine.store %cst, %arg5[%arg8] : memref<1300xf64>
      affine.store %cst, %arg7[%arg8] : memref<1300xf64>
      %1 = affine.load %arg5[%arg8] : memref<1300xf64>
      %2 = affine.load %arg7[%arg8] : memref<1300xf64>
      affine.for %arg9 = 0 to %0 {
        %8 = affine.load %arg3[%arg8, %arg9] : memref<1300x1300xf64>
        %9 = affine.load %arg6[%arg9] : memref<1300xf64>
        %10 = mulf %8, %9 : f64
        %11 = addf %10, %1 : f64
        affine.store %11, %arg5[%arg8] : memref<1300xf64>
        %12 = affine.load %arg4[%arg8, %arg9] : memref<1300x1300xf64>
        %13 = affine.load %arg6[%arg9] : memref<1300xf64>
        %14 = mulf %12, %13 : f64
        %15 = addf %14, %2 : f64
        affine.store %15, %arg7[%arg8] : memref<1300xf64>
      }
      %3 = affine.load %arg5[%arg8] : memref<1300xf64>
      %4 = mulf %arg1, %3 : f64
      %5 = affine.load %arg7[%arg8] : memref<1300xf64>
      %6 = mulf %arg2, %5 : f64
      %7 = addf %4, %6 : f64
      affine.store %7, %arg7[%arg8] : memref<1300xf64>
    }
    return
  }
}
