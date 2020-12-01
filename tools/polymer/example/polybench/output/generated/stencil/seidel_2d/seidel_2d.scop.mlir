#map0 = affine_map<() -> (1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<()[s0] -> (s0)>
#map4 = affine_map<(d0) -> (d0 - 1)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0) -> (d0 + 1)>


module {
  func @kernel_seidel_2d(%arg0: i32, %arg1: i32, %arg2: memref<2000x2000xf64>) {
    %c1 = constant 1 : index
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg3 = 0 to %0 {
      affine.for %arg4 = 1 to #map1()[%1] {
        affine.for %arg5 = 1 to #map1()[%1] {
          call @S0(%arg2, %arg4, %arg5) : (memref<2000x2000xf64>, index, index) -> ()
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<2000x2000xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %0 = affine.apply #map4(%arg1)
    %1 = affine.load %arg0[%0, %arg2] : memref<2000x2000xf64>
    %2 = affine.load %arg0[%arg1, %arg2] : memref<2000x2000xf64>
    %3 = affine.apply #map4(%arg2)
    %4 = affine.load %arg0[%0, %3] : memref<2000x2000xf64>
    %5 = addf %4, %1 : f64
    %6 = affine.load %arg0[%arg1, %3] : memref<2000x2000xf64>
    %7 = affine.apply #map6(%arg1)
    %8 = affine.load %arg0[%7, %3] : memref<2000x2000xf64>
    %9 = affine.load %arg0[%7, %arg2] : memref<2000x2000xf64>
    %10 = affine.apply #map6(%arg2)
    %11 = affine.load %arg0[%0, %10] : memref<2000x2000xf64>
    %12 = addf %5, %11 : f64
    %13 = addf %12, %6 : f64
    %14 = addf %13, %2 : f64
    %15 = affine.load %arg0[%arg1, %10] : memref<2000x2000xf64>
    %16 = addf %14, %15 : f64
    %17 = addf %16, %8 : f64
    %18 = addf %17, %9 : f64
    %19 = affine.load %arg0[%7, %10] : memref<2000x2000xf64>
    %20 = addf %18, %19 : f64
    %cst = constant 9.000000e+00 : f64
    %21 = divf %20, %cst : f64
    affine.store %21, %arg0[%arg1, %arg2] : memref<2000x2000xf64>
    return
  }
}
