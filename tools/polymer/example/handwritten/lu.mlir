#map = affine_map<(d0) -> (d0)>
module  {
  func @kernel_lu(%arg0: i32, %arg1: memref<2000x2000xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg2 = 0 to %0 {
      affine.for %arg3 = 0 to #map(%arg2) {
        %1 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
        affine.for %arg4 = 0 to #map(%arg3) {
          %5 = affine.load %arg1[%arg2, %arg4] : memref<2000x2000xf64>
          %6 = affine.load %arg1[%arg4, %arg3] : memref<2000x2000xf64>
          %7 = mulf %5, %6 : f64
          %8 = subf %1, %7 : f64
          affine.store %8, %arg1[%arg2, %arg3] : memref<2000x2000xf64>
        }
        %2 = affine.load %arg1[%arg3, %arg3] : memref<2000x2000xf64>
        %3 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
        %4 = divf %3, %2 : f64
        affine.store %4, %arg1[%arg2, %arg3] : memref<2000x2000xf64>
      }
      affine.for %arg3 = #map(%arg2) to %0 {
        %1 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
        affine.for %arg4 = 0 to #map(%arg2) {
          %2 = affine.load %arg1[%arg2, %arg4] : memref<2000x2000xf64>
          %3 = affine.load %arg1[%arg4, %arg3] : memref<2000x2000xf64>
          %4 = mulf %2, %3 : f64
          %5 = subf %1, %4 : f64
          affine.store %5, %arg1[%arg2, %arg3] : memref<2000x2000xf64>
        }
      }
    }
    return
  }
}
