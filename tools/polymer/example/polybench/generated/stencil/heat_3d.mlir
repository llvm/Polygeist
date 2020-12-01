#map0 = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0) -> (d0 - 1)>
module  {
  func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<120x120x120xf64>, %arg3: memref<120x120x120xf64>) {
    %cst = constant 1.250000e-01 : f64
    %cst_0 = constant 2.000000e+00 : f64
    %c1 = constant 1 : index
    %0 = index_cast %arg1 : i32 to index
    affine.for %arg4 = 1 to 501 {
      affine.for %arg5 = 1 to #map0()[%0] {
        affine.for %arg6 = 1 to #map0()[%0] {
          affine.for %arg7 = 1 to #map0()[%0] {
            %1 = affine.apply #map1(%arg5)
            %2 = affine.load %arg2[%1, %arg6, %arg7] : memref<120x120x120xf64>
            %3 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
            %4 = mulf %cst_0, %3 : f64
            %5 = subf %2, %4 : f64
            %6 = affine.apply #map2(%arg5)
            %7 = affine.load %arg2[%6, %arg6, %arg7] : memref<120x120x120xf64>
            %8 = addf %5, %7 : f64
            %9 = mulf %cst, %8 : f64
            %10 = affine.apply #map1(%arg6)
            %11 = affine.load %arg2[%arg5, %10, %arg7] : memref<120x120x120xf64>
            %12 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
            %13 = mulf %cst_0, %12 : f64
            %14 = subf %11, %13 : f64
            %15 = affine.apply #map2(%arg6)
            %16 = affine.load %arg2[%arg5, %15, %arg7] : memref<120x120x120xf64>
            %17 = addf %14, %16 : f64
            %18 = mulf %cst, %17 : f64
            %19 = addf %9, %18 : f64
            %20 = affine.apply #map1(%arg7)
            %21 = affine.load %arg2[%arg5, %arg6, %20] : memref<120x120x120xf64>
            %22 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
            %23 = mulf %cst_0, %22 : f64
            %24 = subf %21, %23 : f64
            %25 = affine.apply #map2(%arg7)
            %26 = affine.load %arg2[%arg5, %arg6, %25] : memref<120x120x120xf64>
            %27 = addf %24, %26 : f64
            %28 = mulf %cst, %27 : f64
            %29 = addf %19, %28 : f64
            %30 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
            %31 = addf %29, %30 : f64
            affine.store %31, %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
          }
        }
      }
      affine.for %arg5 = 1 to #map0()[%0] {
        affine.for %arg6 = 1 to #map0()[%0] {
          affine.for %arg7 = 1 to #map0()[%0] {
            %1 = affine.apply #map1(%arg5)
            %2 = affine.load %arg3[%1, %arg6, %arg7] : memref<120x120x120xf64>
            %3 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
            %4 = mulf %cst_0, %3 : f64
            %5 = subf %2, %4 : f64
            %6 = affine.apply #map2(%arg5)
            %7 = affine.load %arg3[%6, %arg6, %arg7] : memref<120x120x120xf64>
            %8 = addf %5, %7 : f64
            %9 = mulf %cst, %8 : f64
            %10 = affine.apply #map1(%arg6)
            %11 = affine.load %arg3[%arg5, %10, %arg7] : memref<120x120x120xf64>
            %12 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
            %13 = mulf %cst_0, %12 : f64
            %14 = subf %11, %13 : f64
            %15 = affine.apply #map2(%arg6)
            %16 = affine.load %arg3[%arg5, %15, %arg7] : memref<120x120x120xf64>
            %17 = addf %14, %16 : f64
            %18 = mulf %cst, %17 : f64
            %19 = addf %9, %18 : f64
            %20 = affine.apply #map1(%arg7)
            %21 = affine.load %arg3[%arg5, %arg6, %20] : memref<120x120x120xf64>
            %22 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
            %23 = mulf %cst_0, %22 : f64
            %24 = subf %21, %23 : f64
            %25 = affine.apply #map2(%arg7)
            %26 = affine.load %arg3[%arg5, %arg6, %25] : memref<120x120x120xf64>
            %27 = addf %24, %26 : f64
            %28 = mulf %cst, %27 : f64
            %29 = addf %19, %28 : f64
            %30 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
            %31 = addf %29, %30 : f64
            affine.store %31, %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
          }
        }
      }
    }
    return
  }
}
