#map0 = affine_map<() -> (1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<()[s0] -> (s0)>
#map4 = affine_map<()[s0, s1] -> (s0, s1)>
#map5 = affine_map<()[s0, s1] -> (s0, s1 - 1)>
#map6 = affine_map<()[s0, s1] -> (s0, s1 + 1)>
#map7 = affine_map<()[s0, s1] -> (s0 + 1, s1)>
#map8 = affine_map<()[s0, s1] -> (s0 - 1, s1)>
#map9 = affine_map<(d0) -> (1, d0 * 32)>
#map10 = affine_map<(d0)[s0] -> (s0 - 1, d0 * 32 + 32)>
#map11 = affine_map<()[s0] -> ((s0 - 2) floordiv 32 + 1)>


module {
  func @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<1300x1300xf64>, %arg3: memref<1300x1300xf64>) {
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
    %0 = affine.load %arg3[symbol(%arg1), symbol(%arg2)] : memref<1300x1300xf64>
    %1 = affine.load %arg3[symbol(%arg1), symbol(%arg2) - 1] : memref<1300x1300xf64>
    %2 = addf %0, %1 : f64
    %3 = affine.load %arg3[symbol(%arg1), symbol(%arg2) + 1] : memref<1300x1300xf64>
    %4 = addf %2, %3 : f64
    %5 = affine.load %arg3[symbol(%arg1) + 1, symbol(%arg2)] : memref<1300x1300xf64>
    %6 = addf %4, %5 : f64
    %7 = affine.load %arg3[symbol(%arg1) - 1, symbol(%arg2)] : memref<1300x1300xf64>
    %8 = addf %6, %7 : f64
    %9 = mulf %cst, %8 : f64
    affine.store %9, %arg0[symbol(%arg1), symbol(%arg2)] : memref<1300x1300xf64>
    return
  }
  func @S1(%arg0: memref<1300x1300xf64>, %arg1: index, %arg2: index, %arg3: memref<1300x1300xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e-01 : f64
    %0 = affine.load %arg3[symbol(%arg1), symbol(%arg2)] : memref<1300x1300xf64>
    %1 = affine.load %arg3[symbol(%arg1), symbol(%arg2) - 1] : memref<1300x1300xf64>
    %2 = addf %0, %1 : f64
    %3 = affine.load %arg3[symbol(%arg1), symbol(%arg2) + 1] : memref<1300x1300xf64>
    %4 = addf %2, %3 : f64
    %5 = affine.load %arg3[symbol(%arg1) + 1, symbol(%arg2)] : memref<1300x1300xf64>
    %6 = addf %4, %5 : f64
    %7 = affine.load %arg3[symbol(%arg1) - 1, symbol(%arg2)] : memref<1300x1300xf64>
    %8 = addf %6, %7 : f64
    %9 = mulf %cst, %8 : f64
    affine.store %9, %arg0[symbol(%arg1), symbol(%arg2)] : memref<1300x1300xf64>
    return
  }
  func @"kernel_jacobi_2d\909\D0\02_new"(%arg0: memref<1300x1300xf64>, %arg1: memref<1300x1300xf64>, %arg2: i32, %arg3: i32) {
    %0 = index_cast %arg3 : i32 to index
    %1 = index_cast %arg2 : i32 to index
    affine.for %arg4 = 0 to %1 {
      affine.for %arg5 = 0 to #map11()[%0] {
        affine.for %arg6 = 0 to #map11()[%0] {
          affine.for %arg7 = max #map9(%arg5) to min #map10(%arg5)[%0] {
            affine.for %arg8 = max #map9(%arg6) to min #map10(%arg6)[%0] {
              call @S0(%arg1, %arg4, %arg7, %arg0) : (memref<1300x1300xf64>, index, index, memref<1300x1300xf64>) -> ()
            }
          }
        }
      }
      affine.for %arg5 = 0 to #map11()[%0] {
        affine.for %arg6 = 0 to #map11()[%0] {
          affine.for %arg7 = max #map9(%arg5) to min #map10(%arg5)[%0] {
            affine.for %arg8 = max #map9(%arg6) to min #map10(%arg6)[%0] {
              call @S1(%arg0, %arg4, %arg7, %arg1) : (memref<1300x1300xf64>, index, index, memref<1300x1300xf64>) -> ()
            }
          }
        }
      }
    }
    return
  }
}
