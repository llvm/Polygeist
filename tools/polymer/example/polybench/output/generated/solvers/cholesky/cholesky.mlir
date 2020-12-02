#map = affine_map<(d0) -> (d0)>
module  {
  func @kernel_cholesky(%arg0: i32, %arg1: memref<2000x2000xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg2 = 0 to %0 {
      affine.for %arg3 = 0 to #map(%arg2) {
        %4 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
        affine.for %arg4 = 0 to #map(%arg3) {
          %8 = affine.load %arg1[%arg2, %arg4] : memref<2000x2000xf64>
          %9 = affine.load %arg1[%arg3, %arg4] : memref<2000x2000xf64>
          %10 = mulf %8, %9 : f64
          %11 = subf %4, %10 : f64
          affine.store %11, %arg1[%arg2, %arg3] : memref<2000x2000xf64>
        }
        %5 = affine.load %arg1[%arg3, %arg3] : memref<2000x2000xf64>
        %6 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
        %7 = divf %6, %5 : f64
        affine.store %7, %arg1[%arg2, %arg3] : memref<2000x2000xf64>
      }
      %1 = affine.load %arg1[%arg2, %arg2] : memref<2000x2000xf64>
      affine.for %arg3 = 0 to #map(%arg2) {
        %4 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
        %5 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
        %6 = mulf %4, %5 : f64
        %7 = subf %1, %6 : f64
        affine.store %7, %arg1[%arg2, %arg2] : memref<2000x2000xf64>
      }
      %2 = affine.load %arg1[%arg2, %arg2] : memref<2000x2000xf64>
      %3 = sqrt %2 : f64
      affine.store %3, %arg1[%arg2, %arg2] : memref<2000x2000xf64>
    }
    return
  }
}

