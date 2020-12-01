#map0 = affine_map<() -> (1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<()[s0] -> (s0)>
#map4 = affine_map<(d0) -> (d0 - 1)>
#map5 = affine_map<(d0) -> (d0)>
#map6 = affine_map<(d0) -> (d0 + 1)>


module {
  func @kernel_jacobi_1d(%arg0: i32, %arg1: i32, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>) {
    %c1 = constant 1 : index
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
    %0 = affine.apply #map4(%arg1)
    %1 = affine.load %arg2[%0] : memref<2000xf64>
    %2 = affine.load %arg2[%arg1] : memref<2000xf64>
    %3 = addf %1, %2 : f64
    %4 = affine.apply #map6(%arg1)
    %5 = affine.load %arg2[%4] : memref<2000xf64>
    %6 = addf %3, %5 : f64
    %7 = mulf %cst, %6 : f64
    affine.store %7, %arg0[%arg1] : memref<2000xf64>
    return
  }
  func @S1(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>) attributes {scop.stmt} {
    %cst = constant 3.333300e-01 : f64
    %0 = affine.apply #map4(%arg1)
    %1 = affine.load %arg2[%0] : memref<2000xf64>
    %2 = affine.load %arg2[%arg1] : memref<2000xf64>
    %3 = addf %1, %2 : f64
    %4 = affine.apply #map6(%arg1)
    %5 = affine.load %arg2[%4] : memref<2000xf64>
    %6 = addf %3, %5 : f64
    %7 = mulf %cst, %6 : f64
    affine.store %7, %arg0[%arg1] : memref<2000xf64>
    return
  }
}
