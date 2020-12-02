module  {
  func @kernel_doitgen(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<150x140x160xf64>, %arg4: memref<160x160xf64>, %arg5: memref<160xf64>) {
    %cst = constant 0.000000e+00 : f64
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    %2 = index_cast %arg2 : i32 to index
    affine.for %arg6 = 0 to %0 {
      affine.for %arg7 = 0 to %1 {
        affine.for %arg8 = 0 to %2 {
          affine.store %cst, %arg5[%arg8] : memref<160xf64>
          %3 = affine.load %arg5[%arg8] : memref<160xf64>
          affine.for %arg9 = 0 to %2 {
            %4 = affine.load %arg3[%arg6, %arg7, %arg9] : memref<150x140x160xf64>
            %5 = affine.load %arg4[%arg9, %arg8] : memref<160x160xf64>
            %6 = mulf %4, %5 : f64
            %7 = addf %3, %6 : f64
            affine.store %7, %arg5[%arg8] : memref<160xf64>
          }
        }
        affine.for %arg8 = 0 to %2 {
          %3 = affine.load %arg5[%arg8] : memref<160xf64>
          affine.store %3, %arg3[%arg6, %arg7, %arg8] : memref<150x140x160xf64>
        }
      }
    }
    return
  }
}

