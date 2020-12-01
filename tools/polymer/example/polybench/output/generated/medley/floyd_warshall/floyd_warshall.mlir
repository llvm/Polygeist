#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<()[s0] -> (s0)>


module {
  func @kernel_floyd_warshall(%arg0: i32, %arg1: memref<2800x2800xi32>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg2 = 0 to %0 {
      affine.for %arg3 = 0 to %0 {
        %1 = affine.load %arg1[%arg3, %arg2] : memref<2800x2800xi32>
        affine.for %arg4 = 0 to %0 {
          %2 = affine.load %arg1[%arg3, %arg4] : memref<2800x2800xi32>
          %3 = affine.load %arg1[%arg2, %arg4] : memref<2800x2800xi32>
          %4 = addi %1, %3 : i32
          %5 = cmpi "slt", %2, %4 : i32
          %6 = scf.if %5 -> (i32) {
            %7 = affine.load %arg1[%arg3, %arg4] : memref<2800x2800xi32>
            scf.yield %7 : i32
          } else {
            %7 = affine.load %arg1[%arg3, %arg2] : memref<2800x2800xi32>
            %8 = affine.load %arg1[%arg2, %arg4] : memref<2800x2800xi32>
            %9 = addi %7, %8 : i32
            scf.yield %9 : i32
          }
          affine.store %6, %arg1[%arg3, %arg4] : memref<2800x2800xi32>
        }
      }
    }
    return
  }
}
