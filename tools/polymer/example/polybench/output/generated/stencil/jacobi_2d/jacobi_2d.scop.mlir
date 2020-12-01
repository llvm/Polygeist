#map0 = affine_map<() -> (1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<()[s0] -> (s0)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0) -> (d0 - 1)>
#map6 = affine_map<(d0) -> (d0 + 1)>


module {
  func @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<1300x1300xf64>, %arg3: memref<1300x1300xf64>) {
    %c1 = constant 1 : index
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg4 = 0 to %0 {
      affine.for %arg5 = 1 to #map1()[%1] {
        affine.for %arg6 = 1 to #map1()[%1] {
          call @S0(%arg3, %arg5, %arg6, %arg2) : (memref<1300x1300xf64>, index, index, memref<1300x1300xf64>) -> ()
        }
      }
      affine.for %arg5 = 1 to #map1()[%1] {
        affine.for %arg6 = 1 to #map1()[%1] {
          call @S1(%arg2, %arg5, %arg6, %arg3) : (memref<1300x1300xf64>, index, index, memref<1300x1300xf64>) -> ()
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<1300x1300xf64>, %arg1: index, %arg2: index, %arg3: memref<1300x1300xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e-01 : f64
    %0 = affine.load %arg3[%arg1, %arg2] : memref<1300x1300xf64>
    %1 = affine.apply #map5(%arg2)
    %2 = affine.load %arg3[%arg1, %1] : memref<1300x1300xf64>
    %3 = addf %0, %2 : f64
    %4 = affine.apply #map6(%arg2)
    %5 = affine.load %arg3[%arg1, %4] : memref<1300x1300xf64>
    %6 = addf %3, %5 : f64
    %7 = affine.apply #map6(%arg1)
    %8 = affine.load %arg3[%7, %arg2] : memref<1300x1300xf64>
    %9 = addf %6, %8 : f64
    %10 = affine.apply #map5(%arg1)
    %11 = affine.load %arg3[%10, %arg2] : memref<1300x1300xf64>
    %12 = addf %9, %11 : f64
    %13 = mulf %cst, %12 : f64
    affine.store %13, %arg0[%arg1, %arg2] : memref<1300x1300xf64>
    return
  }
  func @S1(%arg0: memref<1300x1300xf64>, %arg1: index, %arg2: index, %arg3: memref<1300x1300xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e-01 : f64
    %0 = affine.load %arg3[%arg1, %arg2] : memref<1300x1300xf64>
    %1 = affine.apply #map5(%arg2)
    %2 = affine.load %arg3[%arg1, %1] : memref<1300x1300xf64>
    %3 = addf %0, %2 : f64
    %4 = affine.apply #map6(%arg2)
    %5 = affine.load %arg3[%arg1, %4] : memref<1300x1300xf64>
    %6 = addf %3, %5 : f64
    %7 = affine.apply #map6(%arg1)
    %8 = affine.load %arg3[%7, %arg2] : memref<1300x1300xf64>
    %9 = addf %6, %8 : f64
    %10 = affine.apply #map5(%arg1)
    %11 = affine.load %arg3[%10, %arg2] : memref<1300x1300xf64>
    %12 = addf %9, %11 : f64
    %13 = mulf %cst, %12 : f64
    affine.store %13, %arg0[%arg1, %arg2] : memref<1300x1300xf64>
    return
  }
}
