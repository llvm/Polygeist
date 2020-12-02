#map0 = affine_map<() -> (1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<() -> (501)>
#map3 = affine_map<()[s0, s1, s2] -> (s0 + 1, s1, s2)>
#map4 = affine_map<()[s0, s1, s2] -> (s0, s1, s2)>
#map5 = affine_map<()[s0, s1, s2] -> (s0 - 1, s1, s2)>
#map6 = affine_map<()[s0, s1, s2] -> (s0, s1 + 1, s2)>
#map7 = affine_map<()[s0, s1, s2] -> (s0, s1 - 1, s2)>
#map8 = affine_map<()[s0, s1, s2] -> (s0, s1, s2 + 1)>
#map9 = affine_map<()[s0, s1, s2] -> (s0, s1, s2 - 1)>
#map10 = affine_map<(d0) -> (1, d0 * 32)>
#map11 = affine_map<(d0)[s0] -> (s0 - 1, d0 * 32 + 32)>
#map12 = affine_map<() -> (0)>
#map13 = affine_map<()[s0] -> ((s0 - 2) floordiv 32 + 1)>


module {
  func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<120x120x120xf64>, %arg3: memref<120x120x120xf64>) {
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
    %cst = constant 1.250000e-01 : f64
    %cst_0 = constant 2.000000e+00 : f64
    %0 = affine.load %arg4[symbol(%arg1) + 1, symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %1 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %2 = affine.load %arg4[symbol(%arg1) - 1, symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %3 = affine.load %arg4[symbol(%arg1), symbol(%arg2) + 1, symbol(%arg3)] : memref<120x120x120xf64>
    %4 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %5 = affine.load %arg4[symbol(%arg1), symbol(%arg2) - 1, symbol(%arg3)] : memref<120x120x120xf64>
    %6 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3) + 1] : memref<120x120x120xf64>
    %7 = mulf %cst_0, %1 : f64
    %8 = subf %0, %7 : f64
    %9 = addf %8, %2 : f64
    %10 = mulf %cst, %9 : f64
    %11 = mulf %cst_0, %4 : f64
    %12 = subf %3, %11 : f64
    %13 = addf %12, %5 : f64
    %14 = mulf %cst, %13 : f64
    %15 = addf %10, %14 : f64
    %16 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %17 = mulf %cst_0, %16 : f64
    %18 = subf %6, %17 : f64
    %19 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3) - 1] : memref<120x120x120xf64>
    %20 = addf %18, %19 : f64
    %21 = mulf %cst, %20 : f64
    %22 = addf %15, %21 : f64
    %23 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %24 = addf %22, %23 : f64
    affine.store %24, %arg0[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    return
  }
  func @S1(%arg0: memref<120x120x120xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<120x120x120xf64>) attributes {scop.stmt} {
    %cst = constant 1.250000e-01 : f64
    %cst_0 = constant 2.000000e+00 : f64
    %0 = affine.load %arg4[symbol(%arg1) + 1, symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %1 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %2 = affine.load %arg4[symbol(%arg1) - 1, symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %3 = affine.load %arg4[symbol(%arg1), symbol(%arg2) + 1, symbol(%arg3)] : memref<120x120x120xf64>
    %4 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %5 = affine.load %arg4[symbol(%arg1), symbol(%arg2) - 1, symbol(%arg3)] : memref<120x120x120xf64>
    %6 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3) + 1] : memref<120x120x120xf64>
    %7 = mulf %cst_0, %1 : f64
    %8 = subf %0, %7 : f64
    %9 = addf %8, %2 : f64
    %10 = mulf %cst, %9 : f64
    %11 = mulf %cst_0, %4 : f64
    %12 = subf %3, %11 : f64
    %13 = addf %12, %5 : f64
    %14 = mulf %cst, %13 : f64
    %15 = addf %10, %14 : f64
    %16 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %17 = mulf %cst_0, %16 : f64
    %18 = subf %6, %17 : f64
    %19 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3) - 1] : memref<120x120x120xf64>
    %20 = addf %18, %19 : f64
    %21 = mulf %cst, %20 : f64
    %22 = addf %15, %21 : f64
    %23 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %24 = addf %22, %23 : f64
    affine.store %24, %arg0[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    return
  }
  func @kernel_heat_3d_new(%arg0: memref<120x120x120xf64>, %arg1: memref<120x120x120xf64>, %arg2: i32, %arg3: i32) {
    %0 = index_cast %arg3 : i32 to index
    affine.for %arg4 = 1 to 501 {
      affine.for %arg5 = 0 to #map13()[%0] {
        affine.for %arg6 = 0 to #map13()[%0] {
          affine.for %arg7 = 0 to #map13()[%0] {
            affine.for %arg8 = max #map10(%arg5) to min #map11(%arg5)[%0] {
              affine.for %arg9 = max #map10(%arg6) to min #map11(%arg6)[%0] {
                affine.for %arg10 = max #map10(%arg7) to min #map11(%arg7)[%0] {
                  call @S0(%arg1, %arg4, %arg8, %arg9, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
      }
      affine.for %arg5 = 0 to #map13()[%0] {
        affine.for %arg6 = 0 to #map13()[%0] {
          affine.for %arg7 = 0 to #map13()[%0] {
            affine.for %arg8 = max #map10(%arg5) to min #map11(%arg5)[%0] {
              affine.for %arg9 = max #map10(%arg6) to min #map11(%arg6)[%0] {
                affine.for %arg10 = max #map10(%arg7) to min #map11(%arg7)[%0] {
                  call @S1(%arg0, %arg4, %arg8, %arg9, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
      }
    }
    return
  }
}
