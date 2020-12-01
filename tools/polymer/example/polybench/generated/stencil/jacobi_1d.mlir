#map0 = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> (d0 - 1)>
#map2 = affine_map<(d0) -> (d0 + 1)>
module  {
  func @kernel_jacobi_1d(%arg0: i32, %arg1: i32, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>) {
    %cst = constant 3.333300e-01 : f64
    %c1 = constant 1 : index
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg4 = 0 to %0 {
      affine.for %arg5 = 1 to #map0()[%1] {
        %2 = affine.apply #map1(%arg5)
        %3 = affine.load %arg2[%2] : memref<2000xf64>
        %4 = affine.load %arg2[%arg5] : memref<2000xf64>
        %5 = addf %3, %4 : f64
        %6 = affine.apply #map2(%arg5)
        %7 = affine.load %arg2[%6] : memref<2000xf64>
        %8 = addf %5, %7 : f64
        %9 = mulf %cst, %8 : f64
        affine.store %9, %arg3[%arg5] : memref<2000xf64>
      }
      affine.for %arg5 = 1 to #map0()[%1] {
        %2 = affine.apply #map1(%arg5)
        %3 = affine.load %arg3[%2] : memref<2000xf64>
        %4 = affine.load %arg3[%arg5] : memref<2000xf64>
        %5 = addf %3, %4 : f64
        %6 = affine.apply #map2(%arg5)
        %7 = affine.load %arg3[%6] : memref<2000xf64>
        %8 = addf %5, %7 : f64
        %9 = mulf %cst, %8 : f64
        affine.store %9, %arg2[%arg5] : memref<2000xf64>
      }
    }
    return
  }
}
