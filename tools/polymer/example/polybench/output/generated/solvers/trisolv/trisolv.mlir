#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<(d0) -> (d0, d0)>
#map4 = affine_map<()[s0] -> (s0)>


module {
  func @kernel_trisolv(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg4 = 0 to %0 {
      %1 = affine.load %arg3[%arg4] : memref<2000xf64>
      affine.store %1, %arg2[%arg4] : memref<2000xf64>
      %2 = affine.load %arg2[%arg4] : memref<2000xf64>
      affine.for %arg5 = 0 to #map0(%arg4) {
        %6 = affine.load %arg1[%arg4, %arg5] : memref<2000x2000xf64>
        %7 = affine.load %arg2[%arg5] : memref<2000xf64>
        %8 = mulf %6, %7 : f64
        %9 = subf %2, %8 : f64
        affine.store %9, %arg2[%arg4] : memref<2000xf64>
      }
      %3 = affine.load %arg2[%arg4] : memref<2000xf64>
      %4 = affine.load %arg1[%arg4, %arg4] : memref<2000x2000xf64>
      %5 = divf %3, %4 : f64
      affine.store %5, %arg2[%arg4] : memref<2000xf64>
    }
    return
  }
}
