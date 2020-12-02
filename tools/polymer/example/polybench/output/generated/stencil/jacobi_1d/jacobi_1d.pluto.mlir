#map0 = affine_map<() -> (1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<()[s0] -> (s0)>
#map4 = affine_map<()[s0] -> (s0 + 1)>
#map5 = affine_map<(d0) -> (1, d0 * 32)>
#map6 = affine_map<(d0)[s0] -> (s0 - 1, d0 * 32 + 32)>
#map7 = affine_map<()[s0] -> ((s0 - 2) floordiv 32 + 1)>


module {
  func @kernel_jacobi_1d(%arg0: i32, %arg1: i32, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg4 = 0 to %0 {
      affine.for %arg5 = 1 to #map1()[%1] {
        call @S0(%arg3, %arg5, %arg2) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
      }
      affine.for %arg5 = 1 to #map1()[%1] {
        call @S1(%arg2, %arg5, %arg3) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
      }
    }
    return
  }
  func @S0(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>) attributes {scop.stmt} {
    %cst = constant 3.333300e-01 : f64
    %0 = affine.load %arg2[symbol(%arg1) - 1] : memref<2000xf64>
    %1 = affine.load %arg2[symbol(%arg1)] : memref<2000xf64>
    %2 = addf %0, %1 : f64
    %3 = affine.load %arg2[symbol(%arg1) + 1] : memref<2000xf64>
    %4 = addf %2, %3 : f64
    %5 = mulf %cst, %4 : f64
    affine.store %5, %arg0[symbol(%arg1)] : memref<2000xf64>
    return
  }
  func @S1(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>) attributes {scop.stmt} {
    %cst = constant 3.333300e-01 : f64
    %0 = affine.load %arg2[symbol(%arg1) - 1] : memref<2000xf64>
    %1 = affine.load %arg2[symbol(%arg1)] : memref<2000xf64>
    %2 = addf %0, %1 : f64
    %3 = affine.load %arg2[symbol(%arg1) + 1] : memref<2000xf64>
    %4 = addf %2, %3 : f64
    %5 = mulf %cst, %4 : f64
    affine.store %5, %arg0[symbol(%arg1)] : memref<2000xf64>
    return
  }
  func @"kernel_jacobi_1d@v\1C\02_new"(%arg0: memref<2000xf64>, %arg1: memref<2000xf64>, %arg2: i32, %arg3: i32) {
    %0 = index_cast %arg3 : i32 to index
    %1 = index_cast %arg2 : i32 to index
    affine.for %arg4 = 0 to %1 {
      affine.for %arg5 = 0 to #map7()[%0] {
        affine.for %arg6 = max #map5(%arg5) to min #map6(%arg5)[%0] {
          call @S0(%arg1, %arg4, %arg0) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
      affine.for %arg5 = 0 to #map7()[%0] {
        affine.for %arg6 = max #map5(%arg5) to min #map6(%arg5)[%0] {
          call @S1(%arg0, %arg4, %arg1) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
    }
    return
  }
}
