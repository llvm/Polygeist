#map0 = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> (d0 - 1)>
#map2 = affine_map<(d0) -> (d0 + 1)>
module  {
  func @kernel_seidel_2d(%arg0: i32, %arg1: i32, %arg2: memref<2000x2000xf64>) {
    %cst = constant 9.000000e+00 : f64
    %c1 = constant 1 : index
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg3 = 0 to %0 {
      affine.for %arg4 = 1 to #map0()[%1] {
        affine.for %arg5 = 1 to #map0()[%1] {
          %2 = affine.apply #map1(%arg4)
          %3 = affine.apply #map1(%arg5)
          %4 = affine.load %arg2[%2, %3] : memref<2000x2000xf64>
          %5 = affine.load %arg2[%2, %arg5] : memref<2000x2000xf64>
          %6 = addf %4, %5 : f64
          %7 = affine.apply #map2(%arg5)
          %8 = affine.load %arg2[%2, %7] : memref<2000x2000xf64>
          %9 = addf %6, %8 : f64
          %10 = affine.load %arg2[%arg4, %3] : memref<2000x2000xf64>
          %11 = addf %9, %10 : f64
          %12 = affine.load %arg2[%arg4, %arg5] : memref<2000x2000xf64>
          %13 = addf %11, %12 : f64
          %14 = affine.load %arg2[%arg4, %7] : memref<2000x2000xf64>
          %15 = addf %13, %14 : f64
          %16 = affine.apply #map2(%arg4)
          %17 = affine.load %arg2[%16, %3] : memref<2000x2000xf64>
          %18 = addf %15, %17 : f64
          %19 = affine.load %arg2[%16, %arg5] : memref<2000x2000xf64>
          %20 = addf %18, %19 : f64
          %21 = affine.load %arg2[%16, %7] : memref<2000x2000xf64>
          %22 = addf %20, %21 : f64
          %23 = divf %22, %cst : f64
          affine.store %23, %arg2[%arg4, %arg5] : memref<2000x2000xf64>
        }
      }
    }
    return
  }
}
