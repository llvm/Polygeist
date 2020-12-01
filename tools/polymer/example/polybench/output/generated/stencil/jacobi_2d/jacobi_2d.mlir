#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0 - 1)>
#map2 = affine_map<(d0) -> (d0 + 1)>
#map3 = affine_map<() -> (1)>
#map4 = affine_map<()[s0] -> (s0 - 1)>
#map5 = affine_map<() -> (0)>
#map6 = affine_map<()[s0] -> (s0)>


module {
  func @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<1300x1300xf64>, %arg3: memref<1300x1300xf64>) {
    %cst = constant 2.000000e-01 : f64
    %c1 = constant 1 : index
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg4 = 0 to %0 {
      affine.for %arg5 = 1 to #map4()[%1] {
        affine.for %arg6 = 1 to #map4()[%1] {
          %2 = affine.load %arg2[%arg5, %arg6] : memref<1300x1300xf64>
          %3 = affine.apply #map1(%arg6)
          %4 = affine.load %arg2[%arg5, %3] : memref<1300x1300xf64>
          %5 = addf %2, %4 : f64
          %6 = affine.apply #map2(%arg6)
          %7 = affine.load %arg2[%arg5, %6] : memref<1300x1300xf64>
          %8 = addf %5, %7 : f64
          %9 = affine.apply #map2(%arg5)
          %10 = affine.load %arg2[%9, %arg6] : memref<1300x1300xf64>
          %11 = addf %8, %10 : f64
          %12 = affine.apply #map1(%arg5)
          %13 = affine.load %arg2[%12, %arg6] : memref<1300x1300xf64>
          %14 = addf %11, %13 : f64
          %15 = mulf %cst, %14 : f64
          affine.store %15, %arg3[%arg5, %arg6] : memref<1300x1300xf64>
        }
      }
      affine.for %arg5 = 1 to #map4()[%1] {
        affine.for %arg6 = 1 to #map4()[%1] {
          %2 = affine.load %arg3[%arg5, %arg6] : memref<1300x1300xf64>
          %3 = affine.apply #map1(%arg6)
          %4 = affine.load %arg3[%arg5, %3] : memref<1300x1300xf64>
          %5 = addf %2, %4 : f64
          %6 = affine.apply #map2(%arg6)
          %7 = affine.load %arg3[%arg5, %6] : memref<1300x1300xf64>
          %8 = addf %5, %7 : f64
          %9 = affine.apply #map2(%arg5)
          %10 = affine.load %arg3[%9, %arg6] : memref<1300x1300xf64>
          %11 = addf %8, %10 : f64
          %12 = affine.apply #map1(%arg5)
          %13 = affine.load %arg3[%12, %arg6] : memref<1300x1300xf64>
          %14 = addf %11, %13 : f64
          %15 = mulf %cst, %14 : f64
          affine.store %15, %arg2[%arg5, %arg6] : memref<1300x1300xf64>
        }
      }
    }
    return
  }
}
