#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<()[s0] -> (s0)>


module {
  func @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<2000x2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000xf64>, %arg6: memref<2000xf64>, %arg7: memref<2000xf64>, %arg8: memref<2000xf64>, %arg9: memref<2000xf64>, %arg10: memref<2000xf64>, %arg11: memref<2000xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg12 = 0 to %0 {
      %1 = affine.load %arg4[%arg12] : memref<2000xf64>
      %2 = affine.load %arg6[%arg12] : memref<2000xf64>
      affine.for %arg13 = 0 to %0 {
        %3 = affine.load %arg3[%arg12, %arg13] : memref<2000x2000xf64>
        %4 = affine.load %arg5[%arg13] : memref<2000xf64>
        %5 = mulf %1, %4 : f64
        %6 = addf %3, %5 : f64
        %7 = affine.load %arg7[%arg13] : memref<2000xf64>
        %8 = mulf %2, %7 : f64
        %9 = addf %6, %8 : f64
        affine.store %9, %arg3[%arg12, %arg13] : memref<2000x2000xf64>
      }
    }
    affine.for %arg12 = 0 to %0 {
      %1 = affine.load %arg9[%arg12] : memref<2000xf64>
      affine.for %arg13 = 0 to %0 {
        %2 = affine.load %arg3[%arg13, %arg12] : memref<2000x2000xf64>
        %3 = mulf %arg2, %2 : f64
        %4 = affine.load %arg10[%arg13] : memref<2000xf64>
        %5 = mulf %3, %4 : f64
        %6 = addf %1, %5 : f64
        affine.store %6, %arg9[%arg12] : memref<2000xf64>
      }
    }
    affine.for %arg12 = 0 to %0 {
      %1 = affine.load %arg9[%arg12] : memref<2000xf64>
      %2 = affine.load %arg11[%arg12] : memref<2000xf64>
      %3 = addf %1, %2 : f64
      affine.store %3, %arg9[%arg12] : memref<2000xf64>
    }
    affine.for %arg12 = 0 to %0 {
      %1 = affine.load %arg8[%arg12] : memref<2000xf64>
      affine.for %arg13 = 0 to %0 {
        %2 = affine.load %arg3[%arg12, %arg13] : memref<2000x2000xf64>
        %3 = mulf %arg1, %2 : f64
        %4 = affine.load %arg9[%arg13] : memref<2000xf64>
        %5 = mulf %3, %4 : f64
        %6 = addf %1, %5 : f64
        affine.store %6, %arg8[%arg12] : memref<2000xf64>
      }
    }
    return
  }
}
