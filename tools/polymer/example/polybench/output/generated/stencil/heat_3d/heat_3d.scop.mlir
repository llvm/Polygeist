#map0 = affine_map<() -> (1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<() -> (501)>
#map3 = affine_map<(d0) -> (d0 + 1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0) -> (d0 - 1)>


module {
  func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<120x120x120xf64>, %arg3: memref<120x120x120xf64>) {
    %c1 = constant 1 : index
    %0 = index_cast %arg1 : i32 to index
    affine.for %arg4 = 1 to 501 {
      affine.for %arg5 = 1 to #map1()[%0] {
        affine.for %arg6 = 1 to #map1()[%0] {
          affine.for %arg7 = 1 to #map1()[%0] {
            call @S0(%arg3, %arg5, %arg6, %arg7, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
          }
        }
      }
      affine.for %arg5 = 1 to #map1()[%0] {
        affine.for %arg6 = 1 to #map1()[%0] {
          affine.for %arg7 = 1 to #map1()[%0] {
            call @S1(%arg2, %arg5, %arg6, %arg7, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
          }
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<120x120x120xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<120x120x120xf64>) attributes {scop.stmt} {
    %0 = affine.apply #map3(%arg1)
    %1 = affine.load %arg4[%0, %arg2, %arg3] : memref<120x120x120xf64>
    %2 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %3 = affine.apply #map5(%arg1)
    %4 = affine.load %arg4[%3, %arg2, %arg3] : memref<120x120x120xf64>
    %5 = affine.apply #map3(%arg2)
    %6 = affine.load %arg4[%arg1, %5, %arg3] : memref<120x120x120xf64>
    %7 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %8 = affine.apply #map5(%arg2)
    %9 = affine.load %arg4[%arg1, %8, %arg3] : memref<120x120x120xf64>
    %cst = constant 1.250000e-01 : f64
    %10 = affine.apply #map3(%arg3)
    %11 = affine.load %arg4[%arg1, %arg2, %10] : memref<120x120x120xf64>
    %cst_0 = constant 2.000000e+00 : f64
    %12 = mulf %cst_0, %2 : f64
    %13 = subf %1, %12 : f64
    %14 = addf %13, %4 : f64
    %15 = mulf %cst, %14 : f64
    %16 = mulf %cst_0, %7 : f64
    %17 = subf %6, %16 : f64
    %18 = addf %17, %9 : f64
    %19 = mulf %cst, %18 : f64
    %20 = addf %15, %19 : f64
    %21 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %22 = mulf %cst_0, %21 : f64
    %23 = subf %11, %22 : f64
    %24 = affine.apply #map5(%arg3)
    %25 = affine.load %arg4[%arg1, %arg2, %24] : memref<120x120x120xf64>
    %26 = addf %23, %25 : f64
    %27 = mulf %cst, %26 : f64
    %28 = addf %20, %27 : f64
    %29 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %30 = addf %28, %29 : f64
    affine.store %30, %arg0[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    return
  }
  func @S1(%arg0: memref<120x120x120xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<120x120x120xf64>) attributes {scop.stmt} {
    %0 = affine.apply #map3(%arg1)
    %1 = affine.load %arg4[%0, %arg2, %arg3] : memref<120x120x120xf64>
    %2 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %3 = affine.apply #map5(%arg1)
    %4 = affine.load %arg4[%3, %arg2, %arg3] : memref<120x120x120xf64>
    %5 = affine.apply #map3(%arg2)
    %6 = affine.load %arg4[%arg1, %5, %arg3] : memref<120x120x120xf64>
    %7 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %8 = affine.apply #map5(%arg2)
    %9 = affine.load %arg4[%arg1, %8, %arg3] : memref<120x120x120xf64>
    %cst = constant 1.250000e-01 : f64
    %10 = affine.apply #map3(%arg3)
    %11 = affine.load %arg4[%arg1, %arg2, %10] : memref<120x120x120xf64>
    %cst_0 = constant 2.000000e+00 : f64
    %12 = mulf %cst_0, %2 : f64
    %13 = subf %1, %12 : f64
    %14 = addf %13, %4 : f64
    %15 = mulf %cst, %14 : f64
    %16 = mulf %cst_0, %7 : f64
    %17 = subf %6, %16 : f64
    %18 = addf %17, %9 : f64
    %19 = mulf %cst, %18 : f64
    %20 = addf %15, %19 : f64
    %21 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %22 = mulf %cst_0, %21 : f64
    %23 = subf %11, %22 : f64
    %24 = affine.apply #map5(%arg3)
    %25 = affine.load %arg4[%arg1, %arg2, %24] : memref<120x120x120xf64>
    %26 = addf %23, %25 : f64
    %27 = mulf %cst, %26 : f64
    %28 = addf %20, %27 : f64
    %29 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %30 = addf %28, %29 : f64
    affine.store %30, %arg0[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    return
  }
}
