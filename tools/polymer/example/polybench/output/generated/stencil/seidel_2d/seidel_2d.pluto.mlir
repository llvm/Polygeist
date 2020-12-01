#map0 = affine_map<() -> (1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<()[s0] -> (s0)>
#map4 = affine_map<(d0) -> (d0 - 1)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0) -> (d0 + 1)>
#map7 = affine_map<(d0) -> (d0)>
#map8 = affine_map<(d0, d1) -> (-d0 + d1)>
#map9 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 + d2 + 1)>
#map10 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 + d2 + s0 - 1)>
#map11 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 + 1, d2 * 32 - d1 - s0 + 2)>
#map12 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 32 - d2 + 31, d2 + s0 - 1)>
#map13 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 - d1 * 32, d1 * 32 - s0 + 2, d2 * 16 - s0 + 2, d1 * -32 + d2 * 32 - s0 - 29)>
#map14 = affine_map<(d0, d1, d2)[s0] -> (s0, d0 * 32 + 31, d1 * 16 + 15, d2 * 32 - d0 * 32 + 32, d0 * -32 + d1 * 32 + 31)>
#map15 = affine_map<(d0, d1)[s0] -> ((d0 * 64 - s0 - 28) ceildiv 32, d1)>
#map16 = affine_map<(d0, d1)[s0, s1] -> ((s0 + s1 - 3) floordiv 16 + 1, (d0 * 32 - d1 * 32 + s1 + 29) floordiv 16 + 1, (d0 * 32 + s1 + 60) floordiv 32 + 1, (d1 * 64 + s1 + 59) floordiv 32 + 1, (d1 * 32 + s0 + s1 + 28) floordiv 32 + 1)>
#map17 = affine_map<(d0)[s0] -> (d0 ceildiv 2, (d0 * 32 - s0 + 1) ceildiv 32)>
#map18 = affine_map<(d0)[s0, s1] -> ((s0 + s1 - 3) floordiv 32 + 1, (d0 * 32 + s1 + 29) floordiv 64 + 1, d0 + 1)>
#map19 = affine_map<()[s0, s1] -> ((s0 * 2 + s1 - 4) floordiv 32 + 1)>


module {
  func @kernel_seidel_2d(%arg0: i32, %arg1: i32, %arg2: memref<2000x2000xf64>) {
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
  func @"kernel_seidel_2d\A0\E7Z\02_new"(%arg0: memref<2000x2000xf64>, %arg1: i32, %arg2: i32) {
    %0 = index_cast %arg2 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg3 = 0 to #map19()[%1, %0] {
      affine.for %arg4 = max #map17(%arg3)[%1] to min #map18(%arg3)[%1, %0] {
        affine.for %arg5 = max #map15(%arg3, %arg4)[%0] to min #map16(%arg3, %arg4)[%1, %0] {
          affine.for %arg6 = max #map13(%arg3, %arg4, %arg5)[%0] to min #map14(%arg3, %arg4, %arg5)[%1] {
            affine.for %arg7 = max #map11(%arg4, %arg5, %arg6)[%0] to min #map12(%arg4, %arg5, %arg6)[%0] {
              affine.for %arg8 = max #map9(%arg5, %arg6, %arg7) to min #map10(%arg5, %arg6, %arg7)[%0] {
                %2 = affine.apply #map7(%arg6)
                %3 = affine.apply #map8(%arg6, %arg7)
                call @S0(%arg0, %2, %3) : (memref<2000x2000xf64>, index, index) -> ()
              }
            }
          }
        }
      }
    }
    return
  }
}
