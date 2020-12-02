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
  func @"kernel_seidel_2d\A0\E75\02_new"(%arg0: memref<2000x2000xf64>, %arg1: i32, %arg2: i32) {
    %0 = index_cast %arg2 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg3 = 0 to %1 {
      affine.for %arg4 = 1 to #map1()[%0] {
        affine.for %arg5 = 1 to #map1()[%0] {
          call @S0(%arg0, %arg3, %arg4) : (memref<2000x2000xf64>, index, index) -> ()
        }
      }
    }
    return
  }
}
