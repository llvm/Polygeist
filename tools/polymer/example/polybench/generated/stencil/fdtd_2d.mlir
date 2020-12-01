#map0 = affine_map<(d0) -> (d0 - 1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<(d0) -> (d0 + 1)>
module  {
  func @kernel_fdtd_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<1000x1200xf64>, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1200xf64>, %arg6: memref<500xf64>) {
    %c0 = constant 0 : index
    %cst = constant 5.000000e-01 : f64
    %cst_0 = constant 0.69999999999999996 : f64
    %c1 = constant 1 : index
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg2 : i32 to index
    %2 = index_cast %arg1 : i32 to index
    affine.for %arg7 = 0 to %0 {
      %3 = affine.load %arg6[%arg7] : memref<500xf64>
      affine.for %arg8 = 0 to %1 {
        affine.store %3, %arg4[%c0, %arg8] : memref<1000x1200xf64>
      }
      affine.for %arg8 = 1 to %2 {
        affine.for %arg9 = 0 to %1 {
          %4 = affine.load %arg4[%arg8, %arg9] : memref<1000x1200xf64>
          %5 = affine.load %arg5[%arg8, %arg9] : memref<1000x1200xf64>
          %6 = affine.apply #map0(%arg8)
          %7 = affine.load %arg5[%6, %arg9] : memref<1000x1200xf64>
          %8 = subf %5, %7 : f64
          %9 = mulf %cst, %8 : f64
          %10 = subf %4, %9 : f64
          affine.store %10, %arg4[%arg8, %arg9] : memref<1000x1200xf64>
        }
      }
      affine.for %arg8 = 0 to %2 {
        affine.for %arg9 = 1 to %1 {
          %4 = affine.load %arg3[%arg8, %arg9] : memref<1000x1200xf64>
          %5 = affine.load %arg5[%arg8, %arg9] : memref<1000x1200xf64>
          %6 = affine.apply #map0(%arg9)
          %7 = affine.load %arg5[%arg8, %6] : memref<1000x1200xf64>
          %8 = subf %5, %7 : f64
          %9 = mulf %cst, %8 : f64
          %10 = subf %4, %9 : f64
          affine.store %10, %arg3[%arg8, %arg9] : memref<1000x1200xf64>
        }
      }
      affine.for %arg8 = 0 to #map1()[%2] {
        affine.for %arg9 = 0 to #map1()[%1] {
          %4 = affine.load %arg5[%arg8, %arg9] : memref<1000x1200xf64>
          %5 = affine.apply #map2(%arg9)
          %6 = affine.load %arg3[%arg8, %5] : memref<1000x1200xf64>
          %7 = affine.load %arg3[%arg8, %arg9] : memref<1000x1200xf64>
          %8 = subf %6, %7 : f64
          %9 = affine.apply #map2(%arg8)
          %10 = affine.load %arg4[%9, %arg9] : memref<1000x1200xf64>
          %11 = addf %8, %10 : f64
          %12 = affine.load %arg4[%arg8, %arg9] : memref<1000x1200xf64>
          %13 = subf %11, %12 : f64
          %14 = mulf %cst_0, %13 : f64
          %15 = subf %4, %14 : f64
          affine.store %15, %arg5[%arg8, %arg9] : memref<1000x1200xf64>
        }
      }
    }
    return
  }
}
