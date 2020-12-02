#map0 = affine_map<() -> (1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<()[s0] -> (s0)>
#map4 = affine_map<()[s0, s1] -> (s0 - 1, s1)>
#map5 = affine_map<()[s0, s1] -> (s0, s1)>
#map6 = affine_map<()[s0, s1] -> (s0 - 1, s1 - 1)>
#map7 = affine_map<()[s0, s1] -> (s0, s1 - 1)>
#map8 = affine_map<()[s0, s1] -> (s0 + 1, s1 - 1)>
#map9 = affine_map<()[s0, s1] -> (s0 + 1, s1)>
#map10 = affine_map<()[s0, s1] -> (s0 - 1, s1 + 1)>
#map11 = affine_map<()[s0, s1] -> (s0, s1 + 1)>
#map12 = affine_map<()[s0, s1] -> (s0 + 1, s1 + 1)>
#map13 = affine_map<(d0, d1) -> (-d0 + d1)>
#map14 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 + d2 + 1)>
#map15 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 + d2 + s0 - 1)>
#map16 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 + 1, d2 * 32 - d1 - s0 + 2)>
#map17 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 32 - d2 + 31, d2 + s0 - 1)>
#map18 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 - d1 * 32, d1 * 32 - s0 + 2, d2 * 16 - s0 + 2, d1 * -32 + d2 * 32 - s0 - 29)>
#map19 = affine_map<(d0, d1, d2)[s0] -> (s0, d0 * 32 + 31, d1 * 16 + 15, d2 * 32 - d0 * 32 + 32, d0 * -32 + d1 * 32 + 31)>
#map20 = affine_map<(d0, d1)[s0] -> ((d0 * 64 - s0 - 28) ceildiv 32, d1)>
#map21 = affine_map<(d0, d1)[s0, s1] -> ((s0 + s1 - 3) floordiv 16 + 1, (d0 * 32 - d1 * 32 + s1 + 29) floordiv 16 + 1, (d0 * 32 + s1 + 60) floordiv 32 + 1, (d1 * 64 + s1 + 59) floordiv 32 + 1, (d1 * 32 + s0 + s1 + 28) floordiv 32 + 1)>
#map22 = affine_map<(d0)[s0] -> (d0 ceildiv 2, (d0 * 32 - s0 + 1) ceildiv 32)>
#map23 = affine_map<(d0)[s0, s1] -> ((s0 + s1 - 3) floordiv 32 + 1, (d0 * 32 + s1 + 29) floordiv 64 + 1, d0 + 1)>
#map24 = affine_map<()[s0, s1] -> ((s0 * 2 + s1 - 4) floordiv 32 + 1)>


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
    %cst = constant 9.000000e+00 : f64
    %0 = affine.load %arg0[symbol(%arg1) - 1, symbol(%arg2)] : memref<2000x2000xf64>
    %1 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<2000x2000xf64>
    %2 = affine.load %arg0[symbol(%arg1) - 1, symbol(%arg2) - 1] : memref<2000x2000xf64>
    %3 = addf %2, %0 : f64
    %4 = affine.load %arg0[symbol(%arg1), symbol(%arg2) - 1] : memref<2000x2000xf64>
    %5 = affine.load %arg0[symbol(%arg1) + 1, symbol(%arg2) - 1] : memref<2000x2000xf64>
    %6 = affine.load %arg0[symbol(%arg1) + 1, symbol(%arg2)] : memref<2000x2000xf64>
    %7 = affine.load %arg0[symbol(%arg1) - 1, symbol(%arg2) + 1] : memref<2000x2000xf64>
    %8 = addf %3, %7 : f64
    %9 = addf %8, %4 : f64
    %10 = addf %9, %1 : f64
    %11 = affine.load %arg0[symbol(%arg1), symbol(%arg2) + 1] : memref<2000x2000xf64>
    %12 = addf %10, %11 : f64
    %13 = addf %12, %5 : f64
    %14 = addf %13, %6 : f64
    %15 = affine.load %arg0[symbol(%arg1) + 1, symbol(%arg2) + 1] : memref<2000x2000xf64>
    %16 = addf %14, %15 : f64
    %17 = divf %16, %cst : f64
    affine.store %17, %arg0[symbol(%arg1), symbol(%arg2)] : memref<2000x2000xf64>
    return
  }
  func @kernel_seidel_2d_new(%arg0: memref<2000x2000xf64>, %arg1: i32, %arg2: i32) {
    %0 = index_cast %arg2 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg3 = 0 to #map24()[%1, %0] {
      affine.for %arg4 = max #map22(%arg3)[%1] to min #map23(%arg3)[%1, %0] {
        affine.for %arg5 = max #map20(%arg3, %arg4)[%0] to min #map21(%arg3, %arg4)[%1, %0] {
          affine.for %arg6 = max #map18(%arg3, %arg4, %arg5)[%0] to min #map19(%arg3, %arg4, %arg5)[%1] {
            affine.for %arg7 = max #map16(%arg4, %arg5, %arg6)[%0] to min #map17(%arg4, %arg5, %arg6)[%0] {
              affine.for %arg8 = max #map14(%arg5, %arg6, %arg7) to min #map15(%arg5, %arg6, %arg7)[%0] {
                %2 = affine.apply #map13(%arg6, %arg7)
                call @S0(%arg0, %arg6, %2) : (memref<2000x2000xf64>, index, index) -> ()
              }
            }
          }
        }
      }
    }
    return
  }
}
